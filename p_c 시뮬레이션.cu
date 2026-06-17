/*
 * TFT vs ALLD - CUDA v6
 * nvcc -O3 -arch=sm_86 -std=c++14 -o tft tft_v6.cu
 *
 * 설계 원칙:
 *   - 1 블록 = 1 시뮬레이션
 *   - BT threads가 N개 개체를 분담 (thread i → 개체 i, i+BT, ...)
 *   - 각 thread가 자기 개체의 sc[i], mv[i]만 write → race condition 없음
 *   - 정렬: bitonic sort (global memory, 2의 제곱 N_pad)
 *   - RNG: xorshift32 (레지스터 1개, curand보다 가벼움)
 *   - pc 추정: 로지스틱 fitting (안정화된 Newton-Raphson)
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

// ============================================================
// 모델 파라미터 (논문과 동일)
// ============================================================
#define R_INT 0.05f
#define ALPHA_V 0.005f
#define BETA_V 0.001f
#define EPS2 0.0004f  // (0.02)^2
#define NSIG 0.0008f
#define MSTEP 0.010f
#define MSUB 3
#define BSIG 0.010f

// ============================================================
// 실험 설정
// ============================================================
static const int N_LIST[] = {100,   200,   500,   1000,  2000,  3000,  5000,  7000,
                             10000, 15000, 20000, 30000, 50000, 70000, 100000};
static const int NC = sizeof(N_LIST) / sizeof(int);

static constexpr int MG_DEFAULT = 200;
static constexpr float EARLY_WIN = 0.85f;
static constexpr float EARLY_LOSE = 0.015f;
static constexpr int EARLY_CHECK = 30;
static constexpr int TAIL_LEN = 30;
static constexpr int M_GRID = 12;
static constexpr double TARGET_GB = 5.0;
static constexpr int MAX_SEEDS = 500;
static constexpr int MIN_SEEDS = 20;

__host__ __device__ int getK(int N) {
    return max(2, (int)(0.02f * N));
}
__host__ int getMG(int N) {
    (void)N;
    return MG_DEFAULT;
}
__host__ double getSigK(int N) {
    return 0.762 / sqrt((double)N);
}
__host__ int getSeeds(int N) {
    int NP = 1;
    while (NP < N) NP <<= 1;
    int s = (int)(TARGET_GB * 1e9 / (NP * 21.0 * M_GRID));
    return min(MAX_SEEDS, max(MIN_SEEDS, s));
}

#define CHK(x)                                                                                   \
    do {                                                                                         \
        cudaError_t _e = (x);                                                                    \
        if (_e != cudaSuccess) {                                                                 \
            fprintf(stderr, "CUDA err %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
            exit(1);                                                                             \
        }                                                                                        \
    } while (0)

// ============================================================
// 디바이스 유틸
// ============================================================
__device__ __forceinline__ float refl(float v) {
    if (v < 0.f) return -v;
    if (v > 1.f) return 2.f - v;
    return v;
}
__device__ __forceinline__ unsigned int xshift(unsigned int& s) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
}
__device__ __forceinline__ float xunif(unsigned int& s) {
    return (xshift(s) >> 8) * (1.f / 16777216.f);
}
__device__ __forceinline__ float xnorm(unsigned int& s, float sig) {
    float u = fmaxf(xunif(s), 1e-7f);  // (0,1) 보장: log(0) 및 log(>1)→NaN 방지
    float v = xunif(s);
    return sig * sqrtf(-2.f * logf(u)) * cosf(6.28318530f * v);
}

// ============================================================
// Bitonic Sort  (key 오름차순, idx 동시 정렬)
// 호출자 책임: N_pad는 2의 제곱
// ============================================================
__device__ void bsort(float* key, int* idx, int N_pad, int tid, int bsz) {
    for (int k = 2; k <= N_pad; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < N_pad; i += bsz) {
                int l = i ^ j;
                if (l > i) {
                    bool asc = ((i & k) == 0);
                    bool swap = (asc ? key[i] > key[l] : key[i] < key[l]);
                    if (swap) {
                        float tk = key[i];
                        key[i] = key[l];
                        key[l] = tk;
                        int tv = idx[i];
                        idx[i] = idx[l];
                        idx[l] = tv;
                    }
                }
            }
            __syncthreads();
        }
    }
}

// ============================================================
// 시뮬레이션 커널
//
// 배열 역할:
//   xs  : 위치 (항상 최신)
//   s   : 전략 (char: 1=TFT, 0=ALLD)
//   sc  : 점수 (게임 후 기록, Death-Birth 정렬에 사용)
//   mv  : 이동 벡터 (게임 후 기록)
//   key : 정렬 임시 key (xs 또는 sc 복사본)  ← sc와 완전 분리
//   idx : 정렬 인덱스
//
// thread i 담당: i, i+BT, i+2*BT, ...
// sc[i], mv[i]에만 write → race condition 없음
// key/idx는 bitonic sort 동안 공유되지만 __syncthreads 보장
// ============================================================
#define BT 128

__global__ __launch_bounds__(128, 6) void sim_kernel(int N, int K, int MG, int N_pad, int* d_ntft,
                                                     int n_sims, int seed_off, int* d_res,
                                                     float* d_xs, char* d_s, float* d_sc,
                                                     float* d_mv, float* d_key, int* d_idx) {
    int sim = blockIdx.x;
    if (sim >= n_sims) return;
    int tid = threadIdx.x;
    int bsz = blockDim.x;

    float* xs = d_xs + (long long)sim * N_pad;
    char* s = d_s + (long long)sim * N_pad;
    float* sc = d_sc + (long long)sim * N_pad;
    float* mv = d_mv + (long long)sim * N_pad;
    float* key = d_key + (long long)sim * N_pad;
    int* idx = d_idx + (long long)sim * N_pad;

    // RNG: thread별 독립 상태
    unsigned int rng = (unsigned int)((seed_off + sim) * 1000003u + tid * 2654435761u + 1u);
    for (int w = 0; w < 8; ++w) xshift(rng);

    // ── 초기화 ──────────────────────────────────────────────
    for (int i = tid; i < N_pad; i += bsz) {
        xs[i] = (i < N) ? xunif(rng) : 0.f;
        s[i] = 0;
        sc[i] = 0.f;
        mv[i] = 0.f;
        key[i] = (i < N) ? 0.f : 1e30f;
        idx[i] = i;
    }
    __syncthreads();

    // Fisher-Yates TFT 배치 (thread 0)
    // mv를 int 임시 배열로 재사용 (4바이트 동일)
    if (tid == 0) {
        int ntft = d_ntft[sim];
        int* perm = (int*)mv;
        for (int i = 0; i < N; ++i) perm[i] = i;
        for (int i = 0; i < ntft; ++i) {
            int j = i + (int)(xunif(rng) * (N - i));
            if (j >= N) j = N - 1;
            int t = perm[i];
            perm[i] = perm[j];
            perm[j] = t;
            s[perm[i]] = 1;
        }
        for (int i = 0; i < N; ++i) mv[i] = 0.f;
    }
    __syncthreads();

    // ── shared memory ────────────────────────────────────────
    __shared__ float sh_hist[64];  // >= TAIL_LEN
    __shared__ int sh_done;
    __shared__ float sh_res;
    __shared__ int sh_red[BT];  // reduction buffer

    if (tid == 0) {
        sh_done = 0;
        sh_res = -1.f;
    }
    for (int t = tid; t < 64; t += bsz) sh_hist[t] = 0.f;
    __syncthreads();

    // ── 메인 루프 ────────────────────────────────────────────
    for (int g = 0; g < MG && !sh_done; ++g) {
        // 1. xs 기준 정렬 → key/idx 갱신
        //    (매 세대 실행: Death-Birth가 key를 오염시키므로)
        for (int i = tid; i < N; i += bsz) {
            key[i] = xs[i];
            idx[i] = i;
        }
        for (int i = N + tid; i < N_pad; i += bsz) {
            key[i] = 1e30f;
            idx[i] = i;
        }
        __syncthreads();
        bsort(key, idx, N_pad, tid, bsz);
        // 이후: key[ii]=정렬된 xs, idx[ii]=원래 인덱스

        // 2. 게임+이동
        //    thread ii가 정렬 순서 ii번째 개체 담당
        //    sc[oi], mv[oi]에 write (oi=원래 인덱스)
        //    key는 읽기만 → race condition 없음
        for (int ii = tid; ii < N; ii += bsz) {
            float xi = key[ii];
            int oi = idx[ii];
            int si = (int)(unsigned char)s[oi];

            // 이웃 구간 이진탐색
            int lo = 0;
            {
                int a = 0, b = ii;
                while (a < b) {
                    int m = (a + b) / 2;
                    if (xi - key[m] <= R_INT)
                        b = m;
                    else
                        a = m + 1;
                }
                lo = a;
            }
            int hi = ii;
            {
                int a = ii, b = N - 1;
                while (a < b) {
                    int m = (a + b + 1) / 2;
                    if (key[m] - xi <= R_INT)
                        a = m;
                    else
                        b = m - 1;
                }
                hi = a;
            }

            int nt = 0, na = 0;
            float mvi = 0.f;
            for (int j = lo; j <= hi; ++j) {
                if (j == ii) continue;
                float dx = xi - key[j];
                float pd = dx * dx + EPS2;
                int sj = (int)(unsigned char)s[idx[j]];
                nt += sj;
                na += 1 - sj;
                if (sj)
                    mvi -= ALPHA_V / pd * dx;
                else
                    mvi += BETA_V / pd * dx;
            }
            float deg = (float)(nt + na);
            sc[oi] = si ? 36.f * nt + 11.f * na : 16.f * nt + 12.f * na;
            mv[oi] = (deg > 0.f) ? mvi / deg : 0.f;
        }
        __syncthreads();

        // 3. 이동 적용
        //    thread i가 xs[i]만 write → race condition 없음
        for (int i = tid; i < N; i += bsz) {
            float mvi = mv[i] + xnorm(rng, NSIG);
            mvi = fmaxf(-MSTEP, fminf(MSTEP, mvi));
            float x = xs[i], d = mvi / MSUB;
            for (int sub = 0; sub < MSUB; ++sub) x = refl(x + d);
            xs[i] = x;
        }
        __syncthreads();

        // 4. Death-Birth
        //    key에 sc 복사 후 정렬 (sc는 보존)
        //    idx[0..K-1]=하위K, idx[N-K..N-1]=상위K
        for (int i = tid; i < N; i += bsz) {
            key[i] = sc[i];
            idx[i] = i;
        }
        for (int i = N + tid; i < N_pad; i += bsz) {
            key[i] = 1e30f;
            idx[i] = i;
        }
        __syncthreads();
        bsort(key, idx, N_pad, tid, bsz);

        // 교체: thread ki → ki번째 교체 (dead/par이 모두 다름)
        // dead ∈ [0..K-1], par ∈ [N-K..N-1] → 겹침 없음
        for (int ki = tid; ki < K; ki += bsz) {
            int dead = idx[ki];
            int par = idx[N - K + ki % K];
            s[dead] = s[par];
            xs[dead] = refl(xs[par] + xnorm(rng, BSIG));
        }
        __syncthreads();

        // 5. TFT 집계 (블록 내 리덕션)
        sh_red[tid] = 0;
        for (int i = tid; i < N; i += bsz) sh_red[tid] += (int)(unsigned char)s[i];
        __syncthreads();
        for (int stride = bsz / 2; stride > 0; stride >>= 1) {
            if (tid < stride) sh_red[tid] += sh_red[tid + stride];
            __syncthreads();
        }

        if (tid == 0) {
            int cnt = sh_red[0];
            float ratio = (float)cnt / N;
            sh_hist[g % TAIL_LEN] = ratio;

            if (cnt == 0) {
                sh_res = 0.f;
                sh_done = 1;
            } else if (cnt == N) {
                sh_res = 1.f;
                sh_done = 1;
            } else if (g >= EARLY_CHECK) {
                if (ratio > EARLY_WIN) {
                    sh_res = 1.f;
                    sh_done = 1;
                }
                if (ratio < EARLY_LOSE) {
                    sh_res = 0.f;
                    sh_done = 1;
                }
                if (g >= TAIL_LEN) {
                    float sm = 0.f;
                    for (int t = 0; t < TAIL_LEN; ++t) sm += sh_hist[(g - t + TAIL_LEN) % TAIL_LEN];
                    float mn = sm / TAIL_LEN;
                    if (ratio < 0.05f && ratio < mn * 0.5f) {
                        sh_res = 0.f;
                        sh_done = 1;
                    }
                    if (ratio > 0.70f && ratio > mn * 1.5f) {
                        sh_res = 1.f;
                        sh_done = 1;
                    }
                }
            }
        }
        __syncthreads();
    }

    // 최종 판정
    if (tid == 0) {
        if (sh_res < 0.f) {
            float sm = 0.f;
            for (int t = 0; t < TAIL_LEN; ++t) sm += sh_hist[t];
            sh_res = (sm / TAIL_LEN > (float)d_ntft[sim] / N) ? 1.f : 0.f;
        }
        d_res[sim] = (int)sh_res;
    }
}

// ============================================================
// GPU 버퍼
// ============================================================
int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

struct Bufs {
    int cur_s = 0, cur_NP = 0;
    int* ntft = nullptr;
    int* res = nullptr;
    int* idx = nullptr;
    float* xs = nullptr;
    float* sc = nullptr;
    float* mv = nullptr;
    float* key = nullptr;
    char* s = nullptr;

    void ensure(int ms, int mnp) {
        if (ms <= cur_s && mnp <= cur_NP) return;
        free_all();
        cur_s = ms;
        cur_NP = mnp;
        CHK(cudaMalloc(&ntft, ms * sizeof(int)));
        CHK(cudaMalloc(&res, ms * sizeof(int)));
        long long sz = (long long)ms * mnp;
        CHK(cudaMalloc(&xs, sz * sizeof(float)));
        CHK(cudaMalloc(&s, sz * sizeof(char)));
        CHK(cudaMalloc(&sc, sz * sizeof(float)));
        CHK(cudaMalloc(&mv, sz * sizeof(float)));
        CHK(cudaMalloc(&key, sz * sizeof(float)));
        CHK(cudaMalloc(&idx, sz * sizeof(int)));
        // xs(4)+s(1)+sc(4)+mv(4)+key(4)+idx(4)=21 bytes/elem
        printf("  [GPU] alloc %dseeds x NP=%d = %.0fMB\n", ms, mnp, sz * 21. / 1e6);
    }
    void free_all() {
        if (!ntft) return;
        cudaFree(ntft);
        cudaFree(res);
        cudaFree(xs);
        cudaFree(s);
        cudaFree(sc);
        cudaFree(mv);
        cudaFree(key);
        cudaFree(idx);
        ntft = res = idx = nullptr;
        xs = sc = mv = key = nullptr;
        s = nullptr;
        cur_s = cur_NP = 0;
    }
};

// ============================================================
// run_grid: M_GRID × seeds 시뮬을 한 번에 실행
// ============================================================
void run_grid(int N, int K, int MG, const std::vector<double>& p_grid, std::vector<int>& wins,
              Bufs& b, int seed_off, int seeds) {
    int M = (int)p_grid.size();
    int ns = M * seeds;
    int NP = next_pow2(N);
    b.ensure(ns, NP);

    // ntft 배열 구성
    std::vector<int> h(ns);
    for (int m = 0; m < M; ++m) {
        int ntft = std::max(1, std::min(N - 1, (int)round(p_grid[m] * N)));
        for (int k = 0; k < seeds; ++k) h[m * seeds + k] = ntft;
    }
    CHK(cudaMemcpy(b.ntft, h.data(), ns * sizeof(int), cudaMemcpyHostToDevice));

    sim_kernel<<<ns, BT>>>(N, K, MG, NP, b.ntft, ns, seed_off, b.res, b.xs, b.s, b.sc, b.mv, b.key,
                           b.idx);
    CHK(cudaGetLastError());
    CHK(cudaDeviceSynchronize());

    std::vector<int> hr(ns);
    CHK(cudaMemcpy(hr.data(), b.res, ns * sizeof(int), cudaMemcpyDeviceToHost));

    wins.resize(M);
    for (int m = 0; m < M; ++m) {
        wins[m] = 0;
        for (int k = 0; k < seeds; ++k) wins[m] += hr[m * seeds + k];
    }
}

// ============================================================
// 로지스틱 fitting (안정화된 Newton-Raphson)
// P(win|p) = sigmoid((p - pc) / sigma)
// ============================================================
std::pair<double, double> fit_logistic(const std::vector<double>& pg, const std::vector<int>& wins,
                                       int seeds, double pc0, double sig0) {
    int M = (int)pg.size();
    double pc = pc0;
    double sig = sig0;
    double sig_min = sig0 * 0.02, sig_max = sig0 * 15.0;

    for (int it = 0; it < 300; ++it) {
        double gpc = 0, gs = 0, hpc = 0, hs = 0, hx = 0;
        for (int m = 0; m < M; ++m) {
            double z = (pg[m] - pc) / sig;
            // 수치 안정화: exp(-z) 클램핑
            double ez = exp(std::max(-30.0, std::min(30.0, -z)));
            double mu = 1.0 / (1.0 + ez);
            mu = std::max(1e-7, std::min(1.0 - 1e-7, mu));
            double w = mu * (1.0 - mu);
            double r = wins[m] - seeds * mu;
            gpc += r / sig;
            gs += r * z / sig;
            hpc += seeds * w / (sig * sig);
            hs += seeds * w * z * z / (sig * sig);
            hx += seeds * w * z / (sig * sig);
        }
        double det = hpc * hs - hx * hx;
        if (fabs(det) < 1e-20) break;
        double dpc = (hs * gpc - hx * gs) / det;
        double ds = (-hx * gpc + hpc * gs) / det;
        // 스텝 제한
        dpc = std::max(-sig * 3.0, std::min(sig * 3.0, dpc));
        ds = std::max(-sig * 0.5, std::min(sig * 0.5, ds));
        // (gpc,gs)는 음의 그래디언트(=NLL 그래디언트)이므로 빼야 함.
        // 기존 += 는 부호 반대라 매 반복 오차가 ~2배로 발산 → 경계에 고정됨.
        pc -= dpc;
        sig -= ds;
        pc = std::max(0.001, std::min(0.199, pc));
        sig = std::max(sig_min, std::min(sig_max, sig));
        if (fabs(dpc) < 1e-9 && fabs(ds) < 1e-9) break;
    }

    // 표준오차
    double I = 0.0;
    for (int m = 0; m < M; ++m) {
        double z = (pg[m] - pc) / sig;
        double ez = exp(std::max(-30.0, std::min(30.0, -z)));
        double mu = 1.0 / (1.0 + ez);
        mu = std::max(1e-7, std::min(1.0 - 1e-7, mu));
        I += seeds * mu * (1.0 - mu) / (sig * sig);
    }
    double se = (I > 1e-15) ? 1.0 / sqrt(I) : sig;
    se = std::min(se, 0.05);  // 최대 5%p 클램핑
    return {pc, se};
}

// ============================================================
// 결과 구조체
// ============================================================
struct PcError {
    double pc, se, ci95;
    double lo() const {
        return pc - ci95;
    }
    double hi() const {
        return pc + ci95;
    }
};
struct Result {
    int N, ntft_est;
    double pc, elapsed;
    PcError err;
};

// ============================================================
// find_pc
// ============================================================
Result find_pc(int N, Bufs& b) {
    int K = getK(N);
    int MG = getMG(N);
    int seeds = getSeeds(N);
    double sigK = getSigK(N);
    double pc0 = 1.0 / 21.0;
    auto ti = std::chrono::high_resolution_clock::now();

    printf("  N=%-6d K=%-5d MG=%-4d seeds=%d\n", N, K, MG, seeds);

    // p_grid: pc0 ± 3.5*sigK, [2.5%, 7.5%] 이내
    double margin = std::min(0.025, 3.5 * sigK);
    double lo_p = std::max(0.025, pc0 - margin);
    double hi_p = std::min(0.075, pc0 + margin);
    // 최소 2*sigK 폭 보장
    if (hi_p - lo_p < 2.0 * sigK) {
        lo_p = std::max(0.01, pc0 - sigK);
        hi_p = std::min(0.09, pc0 + sigK);
    }

    std::vector<double> pg(M_GRID);
    for (int m = 0; m < M_GRID; ++m) pg[m] = lo_p + (hi_p - lo_p) * m / (M_GRID - 1);

    std::vector<int> wins;
    run_grid(N, K, MG, pg, wins, b, N * 10000, seeds);

    auto r = fit_logistic(pg, wins, seeds, pc0, sigK);
    double pc = r.first, se = r.second;

    double dt =
        std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - ti).count();
    PcError e{pc, se, 2.0 * se};
    printf("  -> N=%-6d pc=%.4f%% +/-%.4f%%  [%.4f%%, %.4f%%]  %.1fs\n", N, pc * 100, e.ci95 * 100,
           e.lo() * 100, e.hi() * 100, dt);
    return {N, (int)round(pc * N), pc, dt, e};
}

// ============================================================
// Scaling law fitting (3σ 이상치 제외 포함)
// ============================================================
void fit_and_print(const std::vector<int>& Ns, const std::vector<double>& pcs,
                   const std::vector<PcError>& errs) {
    int n = (int)Ns.size();

    // 가우스 소거 (3×3)
    auto solve3 = [](double A[3][4], double coef[3]) {
        for (int col = 0; col < 3; ++col) {
            int p = col;
            for (int r = col + 1; r < 3; ++r)
                if (fabs(A[r][col]) > fabs(A[p][col])) p = r;
            for (int c = 0; c < 4; ++c) std::swap(A[col][c], A[p][c]);
            for (int r = 0; r < 3; ++r) {
                if (r == col) continue;
                double f = A[r][col] / A[col][col];
                for (int c = col; c < 4; ++c) A[r][c] -= f * A[col][c];
            }
        }
        for (int r = 0; r < 3; ++r) coef[r] = A[r][3] / A[r][r];
    };

    auto wls = [&](const std::vector<int>& ns, const std::vector<double>& ps,
                   const std::vector<double>& ws, double coef[3]) {
        double X[3][3] = {}, Xy[3] = {};
        for (int i = 0; i < (int)ns.size(); ++i) {
            double Nv = ns[i], w = ws[i];
            double x[3] = {1., 1. / sqrt(Nv), 1. / Nv};
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) X[r][c] += w * x[r] * x[c];
                Xy[r] += w * x[r] * ps[i];
            }
        }
        double A[3][4];
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) A[r][c] = X[r][c];
            A[r][3] = Xy[r];
        }
        solve3(A, coef);
    };

    // 1단계: OLS rough fit
    std::vector<double> w1(n, 1.0);
    double coef1[3];
    wls(Ns, pcs, w1, coef1);

    // 잔차 & RMSE
    double rmse1 = 0.;
    std::vector<double> res(n);
    for (int i = 0; i < n; ++i) {
        double Nv = Ns[i];
        res[i] = fabs(pcs[i] - (coef1[0] + coef1[1] / sqrt(Nv) + coef1[2] / Nv));
        rmse1 += res[i] * res[i];
    }
    rmse1 = sqrt(rmse1 / n);

    // 2단계: 3σ 이상치 제외
    std::vector<int> Ns2;
    std::vector<double> pcs2, ws2;
    std::vector<bool> excl(n, false);
    for (int i = 0; i < n; ++i) {
        if (res[i] > 3.0 * rmse1) {
            excl[i] = true;
            printf("  [이상치 제외] N=%-7d pc=%8.4f%%  잔차=%8.4f%%p (%.1fσ)\n", Ns[i],
                   pcs[i] * 100, res[i] * 100, res[i] / rmse1);
        } else {
            Ns2.push_back(Ns[i]);
            pcs2.push_back(pcs[i]);
            ws2.push_back(1.0 / (errs[i].ci95 * errs[i].ci95 + 1e-20));
        }
    }
    if ((int)Ns2.size() < 4) {
        printf("  [경고] 이상치 제외 후 데이터 부족, 전체 사용\n");
        Ns2 = Ns;
        pcs2 = pcs;
        ws2.resize(n);
        for (int i = 0; i < n; ++i) ws2[i] = 1.0 / (errs[i].ci95 * errs[i].ci95 + 1e-20);
    }

    // 3단계: WLS
    double coef[3];
    wls(Ns2, pcs2, ws2, coef);

    // 통계
    int n2 = (int)Ns2.size();
    double ym = 0.;
    for (double v : pcs2) ym += v;
    ym /= n2;
    double st = 0., sr = 0.;
    std::vector<double> pred(n);
    for (int i = 0; i < n; ++i) {
        double Nv = Ns[i];
        pred[i] = coef[0] + coef[1] / sqrt(Nv) + coef[2] / Nv;
    }
    for (int i = 0; i < n2; ++i) {
        double Nv = Ns2[i];
        double p = coef[0] + coef[1] / sqrt(Nv) + coef[2] / Nv;
        sr += (pcs2[i] - p) * (pcs2[i] - p);
        st += (pcs2[i] - ym) * (pcs2[i] - ym);
    }
    double r2 = (st > 0) ? 1. - sr / st : 0.;
    double adj_r2 = 1. - (1. - r2) * (n2 - 1.) / (n2 - 4.);
    double rmse = sqrt(sr / n2);
    // 계수 표준오차 (잔차 기반 근사)
    double s2 = sr / (n2 - 3);
    double se_a = sqrt(fabs(s2)) / sqrt((double)n2);
    double se_b = se_a * sqrt((double)Ns2.back());
    double se_c = se_a * (double)Ns2.back();

    printf("\n");
    printf("+---------------------------------------------------------------+\n");
    printf("|    Scaling law:  pc(N) = a + b/sqrt(N) + c/N                |\n");
    if (n2 < n)
        printf("|    (이상치 %d개 제외, 가중 최소자승)                          |\n", n - n2);
    else
        printf("|    (가중 최소자승)                                            |\n");
    printf("+---------------------------------------------------------------+\n\n");
    printf("  계수 (95%% CI):\n");
    printf("    a = pc(∞) = %8.5f%%  ±  %.5f%%\n", coef[0] * 100, 2 * se_a * 100);
    printf("               [%.5f%%, %.5f%%]\n", (coef[0] - 2 * se_a) * 100,
           (coef[0] + 2 * se_a) * 100);
    printf("    b          = %9.5f  ±  %.5f\n", coef[1], 2 * se_b);
    printf("    c          = %9.5f  ±  %.5f\n", coef[2], 2 * se_c);
    printf("\n  적합도 (%d개 점 기준):\n", n2);
    printf("    R²      = %.6f\n", r2);
    printf("    adj-R²  = %.6f\n", adj_r2);
    printf("    RMSE    = %.5f%%p\n", rmse * 100);
    printf("\n  이론값 비교:\n");
    printf("    1/21    = %.5f%%\n", 100. / 21);
    printf("    a       = %.5f%%\n", coef[0] * 100);
    printf("    |diff|  = %.5f%%p  (%.2fσ)\n", fabs(coef[0] - 1. / 21) * 100,
           fabs(coef[0] - 1. / 21) / se_a);
    printf("\n  잔차 분석:\n");
    printf("  +-------------+----------+----------+----------+----------+\n");
    printf("  |      N      |  측정값  |   ±CI95  |  예측값  |   잔차   |\n");
    printf("  +-------------+----------+----------+----------+----------+\n");
    for (int i = 0; i < n; ++i) {
        char f = excl[i] ? 'X' : (fabs(pcs[i] - pred[i]) > 2 * errs[i].ci95 ? '!' : ' ');
        printf("  | %11d | %7.4f%% | %7.4f%% | %7.4f%% | %+7.4f%%p|%c\n", Ns[i], pcs[i] * 100,
               errs[i].ci95 * 100, pred[i] * 100, (pcs[i] - pred[i]) * 100, f);
    }
    printf("  +-------------+----------+----------+----------+----------+\n");
    printf("    X=이상치제외  !=잔차>2σ\n");
}

// ============================================================
// 메인
// ============================================================
int main() {
    cudaDeviceProp prop;
    CHK(cudaGetDeviceProperties(&prop, 0));

    printf("+---------------------------------------------------------------+\n");
    printf("|   TFT vs ALLD  pc(N)  [CUDA v6]                             |\n");
    printf("+---------------------------------------------------------------+\n");
    printf("  GPU : %s  (%.0fGB  %dSM)\n", prop.name, prop.totalGlobalMem / 1e9,
           prop.multiProcessorCount);
    printf("  K=0.02N  |  1/21=%.5f%%\n\n", 100. / 21);

    Bufs b;
    std::vector<Result> results;
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NC; ++i) results.push_back(find_pc(N_LIST[i], b));

    double tot =
        std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();

    // 결과 출력
    printf("\n");
    printf("+---------------------------------------------------------------------------------+\n");
    printf("|              TFT vs ALLD  임계 초기 비율 pc(N)  측정 결과                      |\n");
    printf("|  K=0.02N  R=0.05  (T,R,P,S)=(5,3,1,0)  12라운드  200세대                      |\n");
    printf("|  이론값: 1/21 = 4.76190%%                                                       |\n");
    printf(
        "+---------------------------------------------------------------------------------+\n\n");
    printf("+-------------+-------+------------------------------------------+---------+\n");
    printf("|      N      |   K   |            pc(N)  (95%% CI)               |   sec   |\n");
    printf("+-------------+-------+------------------------------------------+---------+\n");

    std::vector<int> Nr;
    std::vector<double> pr;
    std::vector<PcError> errs;
    for (int i = 0; i < NC; ++i) {
        int N = results[i].N, K = getK(N);
        char flag = (fabs(results[i].pc - 1. / 21) < 0.005) ? '*' : ' ';
        printf("| %11d | %5d |  %7.4f%% +/-%6.4f%%  [%7.4f%%, %7.4f%%] %c| %7.1f |\n", N, K,
               results[i].pc * 100, results[i].err.ci95 * 100, results[i].err.lo() * 100,
               results[i].err.hi() * 100, flag, results[i].elapsed);
        Nr.push_back(N);
        pr.push_back(results[i].pc);
        errs.push_back(results[i].err);
    }
    printf("+-------------+-------+------------------------------------------+---------+\n");
    printf("  * = 1/21 ±0.5%%p 이내\n\n");
    printf("  총 소요: %.1f초 (%.1f분)\n", tot, tot / 60);

    fit_and_print(Nr, pr, errs);

    // CSV
    FILE* f = fopen("pc_summary.csv", "w");
    if (f) {
        fprintf(f, "N,K,K_pct,ntft_est,pc_pct,ci_lo,ci_hi,ci95,elapsed_s\n");
        for (int i = 0; i < NC; ++i) {
            int N = results[i].N, K = getK(N);
            PcError& e = results[i].err;
            fprintf(f, "%d,%d,%.2f,%d,%.5f,%.5f,%.5f,%.5f,%.1f\n", N, K, 100.f * K / N,
                    results[i].ntft_est, results[i].pc * 100, e.lo() * 100, e.hi() * 100,
                    e.ci95 * 100, results[i].elapsed);
        }
        fclose(f);
        printf("\n저장: pc_summary.csv\n");
    }
    printf("+---------------------------------------------------------------+\n");

    b.free_all();
    return 0;
}
