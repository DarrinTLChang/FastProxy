// ============================================================
// PO8e Closed-Loop Trigger (Minimal + Robust) + Optional Local Recording



// just check if nChanis correct and 100




// Current behavior:
// - Fixed FRAME reads (low latency)
// - Feature = multi-channel fast proxy for network bursts
// - Single threshold + refractory trigger logic
// - Sends one TDT-formatted UDP packet (0x55 0xAA header + 1 float)
//   to the RZ device (RZ_IP) using your existing TDTUDP.{h,cpp}
// - Optional timing / backlog diagnostics for online performance checking
//
// Proxy feature implemented here (to match the offline Python code):
// - Select a manual list of channels
// - For each selected channel, run a 350 Hz biquad high-pass filter
//   continuously across frames using persistent filter state
// - For each selected channel, compute NEO over the current FRAME using
//   the last filtered sample from the immediate previous frame
// - Average the NEO values within each channel over the frame
// - Take the median across selected channels -> final feature
//
// Notes on trigger logic:
// - Single threshold plus a refractory period.
// - When feat >= THRESH_PROXY and refractory is inactive, one UDP packet is sent.
// - After sending, the detector enters refractory for REFRACT_MS.
// - During refractory, feature values are still computed, but no trigger can fire.
// - Once refractory ends, if the feature is still high, another UDP packet can be sent.
// - Therefore the feature does NOT need to come back down before re-triggering.
//
// Local recording:
// - Optional raw DET_CH recording (same as before)
// - Optional proxy feature recording: one float32 value per processed FRAME
//
// Important implementation note:
// - The proxy math is written to match the Python code as closely as possible:
//     1) same biquad coefficients
//     2) same continuous filter state per channel
//     3) same use of previous filtered sample from the immediately previous frame
//     4) same NEO formula
//     5) same mean-within-channel then median-across-channels reduction
// ============================================================

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <vector>
#include <string.h>
#include <algorithm>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "Ws2_32.lib")

#include "PO8e.h"
#include "compat.h"
#include "TDTUDP.h"

#define RZ_IP "10.1.0.100"

static std::vector<uint8_t> rawFrame;

// ---------------- USER SETTINGS ----------------
static const double FS_HZ = 24414.0625;   // micro sampling rate
static const int FRAME = 512;             // samples per loop (~5.24 ms)
static const int DET_CH = 1;              // 0-based channel index for optional raw recording

// PO8e data interpretation
static const int FLOAT32_IS_VOLTS = 1;          // Synapse "Unity" float32 usually means VOLTS
static const float UV_PER_COUNT = 0.9765625f;   // only used if bps==2 (int16 counts)

// Proxy feature threshold
static const float THRESH_PROXY = 90.0f;        // edit after inspecting proxy values in your real units

// Refractory after a trigger:
// while refractory is active, no new UDP packet will be sent.
static const int REFRACT_MS = 500;

// UDP payload value (what RZ/UDPRecv will see)
static const float TRIG_VALUE = 1.0f;

// Print feature occasionally
static const int PRINT_FEAT_EVERY_MS = 250;     // print approx every 250 ms

// ---- optional timing / backlog diagnostics ----
static const int PRINT_TIMING_DIAGNOSTICS = 0;  // 1=enable timing print
static const int PRINT_BACKLOG_DIAGNOSTICS = 1; // 1=enable queue/backlog print
static const int PRINT_DIAG_EVERY_MS = 250;     // print diagnostics approx every 250 ms
// ----------------------------------------------

// ---- optional lead-wise CAR (common average reference) ----
// If enabled, each channel is cleaned by subtracting the mean of its
// corresponding 10-channel lead (0–9,10–19,...,90–99).
// This is a memoryless spatial filter applied per sample before HP filtering.
static const int ENABLE_LEAD_CAR = 1;   // 1=enable lead-wise CAR, 0=use raw signals

// ---- proxy channel selection ----
// Enter the channels of interest manually here.
// This replaces the Python --channels parsing.
static const int PROXY_CHANNELS[] = {
    3,5,6,7,21,22,23,24,25,27,81, 82,84,85,86,87//63,64,66,67,68
};
static const int N_PROXY_CH = (int)(sizeof(PROXY_CHANNELS) / sizeof(PROXY_CHANNELS[0]));

// Biquad HP coefficients: 350 Hz, fs=24414.0625, order 2, a0=1
static const double B0 = 0.938290861982229;
static const double B1 = -1.876581723964458;
static const double B2 = 0.938290861982229;
static const double A1 = -1.872770074080853;
static const double A2 = 0.880393373848063;

// Biquad LP coefficients: 2950 Hz, fs=24414.0625, order 2, a0=1
static const double LP_B0 = 0.092356639761006;
static const double LP_B1 = 0.184713279522012;
static const double LP_B2 = 0.092356639761006;
static const double LP_A1 = -0.975802281994301;
static const double LP_A2 = 0.345228841038325;

// ---- local recording ----
static const int  RECORD_DET_CH = 1;                         // 1=enable raw DET_CH recording
static const char RECORD_PATH[] = "C:\\TDT\\det_ch_record.bin";

static const int  RECORD_PROXY_FEATURE = 1;                  // 1=enable proxy feature recording
static const char PROXY_RECORD_PATH[] = "C:\\TDT\\proxy_feature_record3.bin";
// For proxy recording: file contains one float32 feature value per processed FRAME
// ------------------------------------------------

// ---------- helpers ----------
static inline int isFiniteFloat(float x) { return _finite(x) != 0; }

static inline float read_as_float32(const uint8_t* ptr)
{
    float f; memcpy(&f, ptr, 4); return f;
}

static inline int16_t read_as_int16_raw(const uint8_t* ptr)
{
    int16_t v; memcpy(&v, ptr, 2); return v;
}

// high-resolution timestamp in milliseconds using Windows performance counter
static inline double now_ms()
{
    static LARGE_INTEGER freq;
    static int initialized = 0;
    if (!initialized) {
        QueryPerformanceFrequency(&freq);
        initialized = 1;
    }

    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return 1000.0 * (double)counter.QuadPart / (double)freq.QuadPart;
}

// PO8e readBlock() is CHANNEL-MAJOR:
// raw layout: [ch0 samples][ch1 samples]...[chN samples]
static inline const uint8_t* sample_ptr_channel_major(
    const std::vector<uint8_t>& rawBytes,
    int bps, int iSamples,
    int ch, int i)
{
    size_t idx = ((size_t)ch * (size_t)iSamples + (size_t)i) * (size_t)bps;
    return rawBytes.data() + idx;
}

static inline float sample_to_stream_units(const uint8_t* p, int bps)
{
    if (bps == 2)
    {
        int16_t c = read_as_int16_raw(p);
        return (float)c * UV_PER_COUNT; // int16 path converted to uV
    }
    else if (bps == 4)
    {
        float v = read_as_float32(p);
        if (FLOAT32_IS_VOLTS) return v * 1e6f; // V -> uV
        return v;                               // already uV
    }
    return NAN;
}

// Biquad state for one proxy channel.
// This exactly mirrors the Python BiquadHP object state.
struct BiquadState
{
    double hp_x1, hp_x2, hp_y1, hp_y2;
    double lp_x1, lp_x2, lp_y1, lp_y2;
};

static inline void reset_biquad_state(BiquadState& st)
{
    st.hp_x1 = 0.0; st.hp_x2 = 0.0; st.hp_y1 = 0.0; st.hp_y2 = 0.0;
    st.lp_x1 = 0.0; st.lp_x2 = 0.0; st.lp_y1 = 0.0; st.lp_y2 = 0.0;
}

// Process one FRAME for one channel:
// - read samples from the raw channel-major frame
// - convert to stream units (uV in the current configuration)
// - run the biquad HP sample-by-sample
// - write filtered output into filtOut[0..FRAME-1]
// - persistent state is updated in-place
static inline void biquad_bp_process_frame_from_raw(
    const std::vector<uint8_t>& rawBytes,
    int bps,
    int ch,
    int nChan,
    BiquadState& st,
    float* filtOut)
{
    double hx1 = st.hp_x1, hx2 = st.hp_x2, hy1 = st.hp_y1, hy2 = st.hp_y2;
    double lx1 = st.lp_x1, lx2 = st.lp_x2, ly1 = st.lp_y1, ly2 = st.lp_y2;

    for (int i = 0; i < FRAME; i++)
    {
        const uint8_t* p = sample_ptr_channel_major(rawBytes, bps, FRAME, ch, i);
        double x = (double)sample_to_stream_units(p, bps);

        if (ENABLE_LEAD_CAR)
        {
            int leadStart = (ch / 10) * 10;
            int leadEnd = leadStart + 10;
            if (leadEnd > nChan) leadEnd = nChan;

            double sum = 0.0;
            int cnt = 0;
            for (int lc = leadStart; lc < leadEnd; lc++)
            {
                const uint8_t* lp = sample_ptr_channel_major(rawBytes, bps, FRAME, lc, i);
                sum += (double)sample_to_stream_units(lp, bps);
                cnt++;
            }
            x -= sum / (double)cnt;
        }

        // HP 350 Hz
        double hp = B0 * x + B1 * hx1 + B2 * hx2 - A1 * hy1 - A2 * hy2;
        hx2 = hx1; hx1 = x;
        hy2 = hy1; hy1 = hp;

        // LP 2950 Hz
        double lp = LP_B0 * hp + LP_B1 * lx1 + LP_B2 * lx2 - LP_A1 * ly1 - LP_A2 * ly2;
        lx2 = lx1; lx1 = hp;
        ly2 = ly1; ly1 = lp;

        filtOut[i] = (float)lp;
    }

    st.hp_x1 = hx1; st.hp_x2 = hx2; st.hp_y1 = hy1; st.hp_y2 = hy2;
    st.lp_x1 = lx1; st.lp_x2 = lx2; st.lp_y1 = ly1; st.lp_y2 = ly2;
}

// Mean NEO over one FRAME for one channel.
// This matches the Python logic:
//   ext[0] = previous filtered sample from the immediate previous frame
//   ext[1:] = filtered samples in the current frame
//   neo = ext[1:-1]^2 - ext[:-2] * ext[2:]
// which produces FRAME-1 NEO values.
static inline float mean_neo_over_frame(const float* filt, float prevFilteredSample)
{
    if (FRAME < 2) return NAN;

    double s = 0.0;

    // First NEO term uses the previous filtered sample from the immediate previous frame
    // and the first two filtered samples of the current frame.
    {
        double x_prev = (double)prevFilteredSample;
        double x0 = (double)filt[0];
        double x1 = (double)filt[1];
        s += x0 * x0 - x_prev * x1;
    }

    // Remaining terms use triplets entirely within the current frame.
    for (int i = 1; i < FRAME - 1; i++)
    {
        double xm1 = (double)filt[i - 1];
        double x0 = (double)filt[i];
        double xp1 = (double)filt[i + 1];
        s += x0 * x0 - xm1 * xp1;
    }

    return (float)(s / (double)(FRAME - 1));
}

// Median helper that matches NumPy-style median behavior:
// - odd count: middle value
// - even count: average of the two middle values
// The input array is modified in-place.
static inline float median_inplace(float* vals, int n)
{
    if (n <= 0) return NAN;

    int mid = n / 2;
    std::nth_element(vals, vals + mid, vals + n);
    float upper = vals[mid];

    if ((n & 1) != 0)
        return upper;

    std::nth_element(vals, vals + (mid - 1), vals + mid);
    float lower = vals[mid - 1];
    return 0.5f * (lower + upper);
}

// Top-level proxy computation for one FRAME.
// This is the main replacement for the old RMS feature.
//
// For each selected channel:
//   1) filter current FRAME with persistent biquad state
//   2) compute mean NEO over the FRAME using previous-frame continuity
//   3) update previous filtered sample for the next FRAME
// Then take the median across channels.
static inline float compute_proxy_feature_over_frame(
    const std::vector<uint8_t>& rawBytes,
    int bps,
    int nChan,
    const int* proxyChannels,
    int nProxyCh,
    BiquadState* filterStates,
    float* prevFilteredLast,
    float* filtTmp,
    float* chVals)
{
    for (int c = 0; c < nProxyCh; c++)
    {
        int ch = proxyChannels[c];

        biquad_bp_process_frame_from_raw(rawBytes, bps, ch, nChan, filterStates[c], filtTmp);
        chVals[c] = mean_neo_over_frame(filtTmp, prevFilteredLast[c]);
        prevFilteredLast[c] = filtTmp[FRAME - 1];
    }

    return median_inplace(chVals, nProxyCh);
}

int main(int argc, char** argv)
{
    (void)argc; (void)argv;

    // ---- Winsock init ----
    WSADATA wsaData;
    int wsa = WSAStartup(MAKEWORD(2, 2), &wsaData);
    if (wsa != 0) {
        printf("WSAStartup failed with error %d\n", wsa);
        return 1;
    }

    // ---- UDP socket to RZ ----
    printf("Setting up UDP Socket...");
    SOCKET sock = openSocket(inet_addr(RZ_IP));
    if (sock == INVALID_SOCKET) {
        printf(" Could not open socket: %d\n", WSAGetLastError());
        WSACleanup();
        return 1;
    }
    printf("OK.\n");

    if (!checkRZ(sock)) {
        printf("No RZ found at IP: %s.\n", RZ_IP);
        disconnectRZ(sock);
        WSACleanup();
        return 1;
    }
    printf("Found RZ at IP: %s.\n", RZ_IP);

    if (!setRemoteIp(sock)) {
        printf("Failed to point the RZ to this IP.\n");
        disconnectRZ(sock);
        WSACleanup();
        return 1;
    }
    printf("Pointed the RZ to this IP.\n");

    // ---- Card discovery ----
    int total = PO8e::cardCount();
    printf("Found %d PO8e card(s).\n", total);
    if (total <= 0) {
        disconnectRZ(sock);
        WSACleanup();
        return 0;
    }

    const int PRINT_EVERY_FRAMES = (int)((PRINT_FEAT_EVERY_MS * FS_HZ) / ((double)FRAME * 1000.0));
    const int DIAG_EVERY_FRAMES = (int)((PRINT_DIAG_EVERY_MS * FS_HZ) / ((double)FRAME * 1000.0));
    const int printEvery = (PRINT_EVERY_FRAMES < 1) ? 1 : PRINT_EVERY_FRAMES;
    const int diagEvery = (DIAG_EVERY_FRAMES < 1) ? 1 : DIAG_EVERY_FRAMES;

    printf("Settings:\n");
    printf("  FS=%.4f Hz, FRAME=%d (%.4f ms)\n", FS_HZ, FRAME, 1000.0 * (double)FRAME / FS_HZ);
    printf("  DET_CH=%d\n", DET_CH);
    printf("  THRESH_PROXY=%.6f (single threshold + refractory)\n", THRESH_PROXY);
    printf("  REFRACT=%d ms\n", REFRACT_MS);
    printf("  ENABLE_LEAD_CAR=%d\n", ENABLE_LEAD_CAR);
    printf("  N_PROXY_CH=%d\n", N_PROXY_CH);
    printf("  RECORD_DET_CH=%d, RECORD_PATH=%s\n", RECORD_DET_CH, RECORD_PATH);
    printf("  RECORD_PROXY_FEATURE=%d, PROXY_RECORD_PATH=%s\n", RECORD_PROXY_FEATURE, PROXY_RECORD_PATH);
    printf("  PRINT_TIMING_DIAGNOSTICS=%d\n", PRINT_TIMING_DIAGNOSTICS);
    printf("  PRINT_BACKLOG_DIAGNOSTICS=%d\n", PRINT_BACKLOG_DIAGNOSTICS);

    // ---- Connect + stream loop (reconnect on stop) ----
    while (true)
    {
        printf("\nConnecting to PO8e card...\n");
        PO8e* card = PO8e::connectToCard(0);
        if (!card) {
            printf("Connection failed. Retrying...\n");
            Sleep(500);
            continue;
        }
        printf("Connected: %p\n", (void*)card);

        if (!card->startCollecting()) {
            printf("startCollecting() failed: %d\n", card->getLastError());
            PO8e::releaseCard(card);
            Sleep(500);
            continue;
        }

        // fast mode for lower latency
        card->setFastModeEnabled();

        printf("Collecting. Waiting for stream...\n");
        while (card->samplesReady() == 0)
            compatUSleep(2000);

        int nChan = card->numChannels();
        int bps = card->dataSampleSize();
        printf("Stream started. nChan=%d, bps=%d\n", nChan, bps);

        if (DET_CH < 0 || DET_CH >= nChan) {
            printf("ERROR: DET_CH=%d out of range (nChan=%d)\n", DET_CH, nChan);
            PO8e::releaseCard(card);
            break;
        }
        if (!(bps == 2 || bps == 4)) {
            printf("ERROR: Unsupported bps=%d (expected 2 or 4)\n", bps);
            PO8e::releaseCard(card);
            break;
        }
        for (int i = 0; i < N_PROXY_CH; i++) {
            if (PROXY_CHANNELS[i] < 0 || PROXY_CHANNELS[i] >= nChan) {
                printf("ERROR: PROXY_CHANNELS[%d]=%d out of range (nChan=%d)\n", i, PROXY_CHANNELS[i], nChan);
                PO8e::releaseCard(card);
                return 1;
            }
        }

        rawFrame.resize((size_t)FRAME * (size_t)nChan * (size_t)bps);

        // Proxy persistent state for this streaming session.
        // These must persist across consecutive frames to match the Python code.
        std::vector<BiquadState> proxyFilterStates(N_PROXY_CH);
        std::vector<float> proxyPrevFilteredLast(N_PROXY_CH, 0.0f);
        std::vector<float> proxyFiltTmp(FRAME);
        std::vector<float> proxyChVals(N_PROXY_CH);
        for (int i = 0; i < N_PROXY_CH; i++)
            reset_biquad_state(proxyFilterStates[i]);

        // ---- open record file(s) for this streaming session ----
        FILE* fDet = nullptr;
        if (RECORD_DET_CH) {
            fDet = fopen(RECORD_PATH, "ab");
            if (!fDet) {
                printf("ERROR: Could not open record file: %s (raw recording disabled)\n", RECORD_PATH);
            }
            else {
                if (bps == 4) {
                    printf("Recording DET_CH=%d to %s as float32 (bps=4). Units: %s\n",
                        DET_CH, RECORD_PATH, (FLOAT32_IS_VOLTS ? "Volts" : "as-is"));
                }
                else {
                    printf("Recording DET_CH=%d to %s as float32 uV (converted from int16 counts)\n",
                        DET_CH, RECORD_PATH);
                }
            }
        }

        FILE* fProxy = nullptr;
        if (RECORD_PROXY_FEATURE) {
            fProxy = fopen(PROXY_RECORD_PATH, "ab");
            if (!fProxy) {
                printf("ERROR: Could not open proxy feature file: %s (proxy recording disabled)\n", PROXY_RECORD_PATH);
            }
            else {
                printf("Recording proxy feature to %s as float32 (one value per FRAME)\n", PROXY_RECORD_PATH);
            }
        }

        bool stopped = false;
        int frameCount = 0;
        const int REFRACT_FRAMES = (int)((REFRACT_MS * FS_HZ) / ((double)FRAME * 1000.0));
        int refract = 0; // refractory countdown in frames

        // temp buffer for bps==2 raw DET_CH recording (uV float32)
        std::vector<float> recTmp;
        if (bps == 2) recTmp.resize(FRAME);

        // main loop
        while (!stopped)
        {
            if (!card->waitForDataReady())
            {
                // Synapse stop usually causes this
                printf("waitForDataReady() failed -> reconnecting\n");
                break;
            }

            size_t ready = card->samplesReady(&stopped);
            if (stopped) break;

            while (ready >= (size_t)FRAME)
            {
                double t0_total = 0.0;
                double t1_read = 0.0;
                double t2_record = 0.0;
                double t3_feat = 0.0;
                double t4_trigger = 0.0;
                double t5_flush = 0.0;

                if (PRINT_TIMING_DIAGNOSTICS) t0_total = now_ms();

                int got = card->readBlock(rawFrame.data(), FRAME);
                if (PRINT_TIMING_DIAGNOSTICS) t1_read = now_ms();
                if (got != FRAME)
                    break;

                // ---- optional raw DET_CH recording ----
                if (fDet)
                {
                    if (bps == 4)
                    {
                        // Fast path: channel-major means DET_CH samples are contiguous
                        const uint8_t* base = sample_ptr_channel_major(rawFrame, bps, FRAME, DET_CH, 0);
                        size_t wrote = fwrite(base, 4, FRAME, fDet);
                        if (wrote != (size_t)FRAME) {
                            printf("WARNING: fwrite short write (%zu/%d)\n", wrote, FRAME);
                        }
                    }
                    else // bps == 2
                    {
                        for (int i = 0; i < FRAME; i++) {
                            const uint8_t* p = sample_ptr_channel_major(rawFrame, bps, FRAME, DET_CH, i);
                            recTmp[i] = sample_to_stream_units(p, bps); // uV float
                        }
                        size_t wrote = fwrite(recTmp.data(), 4, FRAME, fDet);
                        if (wrote != (size_t)FRAME) {
                            printf("WARNING: fwrite short write (%zu/%d)\n", wrote, FRAME);
                        }
                    }
                }
                if (PRINT_TIMING_DIAGNOSTICS) t2_record = now_ms();

                // ---- feature ----
                float feat = compute_proxy_feature_over_frame(
                    rawFrame,
                    bps,
                    nChan,
                    PROXY_CHANNELS,
                    N_PROXY_CH,
                    proxyFilterStates.data(),
                    proxyPrevFilteredLast.data(),
                    proxyFiltTmp.data(),
                    proxyChVals.data());

                // Optional proxy feature recording
                if (fProxy) {
                    size_t wrote = fwrite(&feat, sizeof(float), 1, fProxy);
                    if (wrote != 1) {
                        printf("WARNING: proxy fwrite short write (%zu/1)\n", wrote);
                    }
                }
                if (PRINT_TIMING_DIAGNOSTICS) t3_feat = now_ms();

                // occasional feature print
                frameCount++;
                if ((frameCount % printEvery) == 0 && isFiniteFloat(feat))
                    printf("proxy_feat=%.6f\n", feat);

                // Trigger logic:
                // send one UDP packet when feat is above threshold and refractory is inactive.
                // After a trigger, enter refractory so stimulation can run for its fixed duration.
                // Once refractory ends, if feat is still high, another trigger can be sent.
                if (refract > 0) refract--;

                if (isFiniteFloat(feat) && refract == 0)
                {
                    if (feat >= THRESH_PROXY)
                    {
                        int ok = sendPacket(sock, TRIG_VALUE);
                        printf("[TRIG] FIRE proxy_feat=%.6f >= %.6f  sendPacket=%d\n",
                            feat, THRESH_PROXY, ok);
                        refract = REFRACT_FRAMES;
                    }
                }
                if (PRINT_TIMING_DIAGNOSTICS) t4_trigger = now_ms();

                // flush exactly FRAME samples (per channel)
                card->flushBufferedData(FRAME);
                if (PRINT_TIMING_DIAGNOSTICS) t5_flush = now_ms();

                ready = card->samplesReady(&stopped);
                if (stopped) break;

                // Optional diagnostics:
                // - total processing time per frame
                // - current backlog in number of samples / frames waiting in queue
                // A backlog larger than one frame means the system is accumulating data.
                if (((PRINT_TIMING_DIAGNOSTICS || PRINT_BACKLOG_DIAGNOSTICS) != 0) && ((frameCount % diagEvery) == 0))
                {
                    if (PRINT_TIMING_DIAGNOSTICS)
                    {
                        double dt_read = t1_read - t0_total;
                        double dt_record = t2_record - t1_read;
                        double dt_feat = t3_feat - t2_record;
                        double dt_trigger = t4_trigger - t3_feat;
                        double dt_flush = t5_flush - t4_trigger;
                        double dt_total = t5_flush - t0_total;
                        double frame_budget_ms = 1000.0 * (double)FRAME / FS_HZ;

                        printf("[TIMING] read=%.3f ms, record=%.3f ms, feat=%.3f ms, trig=%.3f ms, flush=%.3f ms, total=%.3f ms, budget=%.3f ms\n",
                            dt_read, dt_record, dt_feat, dt_trigger, dt_flush, dt_total, frame_budget_ms);
                    }

                    if (PRINT_BACKLOG_DIAGNOSTICS)
                    {
                        size_t backlogSamples = ready;
                        double backlogFrames = (double)backlogSamples / (double)FRAME;
                        int behind = (backlogSamples >= (size_t)(2 * FRAME)) ? 1 : 0;

                        printf("[BACKLOG] ready_samples=%zu, approx_ready_frames=%.2f, behind=%d\n",
                            backlogSamples, backlogFrames, behind);
                    }
                }
            }
        }

        // close record file(s) for this session
        if (fDet) {
            fflush(fDet);
            fclose(fDet);
            fDet = nullptr;
            printf("Raw recording file closed.\n");
        }
        if (fProxy) {
            fflush(fProxy);
            fclose(fProxy);
            fProxy = nullptr;
            printf("Proxy feature file closed.\n");
        }

        printf("Releasing card.\n");
        PO8e::releaseCard(card);
        Sleep(300);
    }

    disconnectRZ(sock);
    WSACleanup();
    return 0;
}
