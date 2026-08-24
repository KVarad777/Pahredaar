/**
 * =============================================================================
 * PROJECT AEGIS: HIGH-PERFORMANCE PAYMENT ROUTER SIMULATOR (simulator.cpp)
 * Mastercard Innovation Challenge @ Global Fintech Fest (GFF) 2026
 * =============================================================================
 * 
 * High-speed, low-latency C++ transaction streaming engine that simulates live
 * payment gateway traffic into the Python FastAPI ML defense pipeline.
 * 
 * Features:
 *   1. Zero-dependency standard POSIX socket networking (Linux/macOS/BSD)
 *      with seamless cross-platform fallback (Windows Winsock2).
 *   2. Fast zero-copy/low-alloc CSV parser with dynamic header mapping.
 *   3. Native high-throughput JSON string serializer (no third-party libs).
 *   4. Chronological timestamp delta pacing with configurable speed multiplier
 *      (1x real-time, 100x accelerated, MAX raw throughput stress-test).
 *   5. Socket resiliency & reconnect loop (auto-retries every 2 seconds).
 *   6. Live ANSI terminal dashboard with real-time TPS, microsecond write latency,
 *      fraud ratio counters, and last payload inspector.
 * 
 * Compilation:
 *   Linux:   g++ -O3 -std=c++17 simulator.cpp -o simulator -pthread
 *   macOS:   clang++ -O3 -std=c++17 simulator.cpp -o simulator
 *   Windows: g++ -O3 -std=c++17 simulator.cpp -o simulator.exe -lws2_32
 * =============================================================================
 */

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <csignal>
#include <algorithm>
#include <atomic>
#include <ctime>

// =============================================================================
// PLATFORM-SPECIFIC SOCKET HEADERS & PORTABILITY WRAPPERS
// =============================================================================
#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #include <windows.h>
    #pragma comment(lib, "ws2_32.lib")

    using socket_t = SOCKET;
    #define IS_INVALID_SOCKET(s) ((s) == INVALID_SOCKET)
    #define CLOSE_SOCKET(s) closesocket(s)
    #define GET_SOCKET_ERR() WSAGetLastError()

    inline void portable_sleep_micros(int64_t micros) {
        if (micros <= 0) return;
        if (micros >= 1000) {
            ::Sleep(static_cast<DWORD>(micros / 1000));
        } else {
            auto start = std::chrono::high_resolution_clock::now();
            while (std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::high_resolution_clock::now() - start).count() < micros) {
                // spin wait for sub-millisecond precision
            }
        }
    }

    inline void portable_sleep_seconds(int sec) {
        if (sec <= 0) return;
        ::Sleep(static_cast<DWORD>(sec * 1000));
    }

    inline void portable_yield() {
        ::Sleep(0);
    }
#else
    #include <sys/types.h>
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <netinet/tcp.h>
    #include <arpa/inet.h>
    #include <netdb.h>
    #include <unistd.h>
    #include <fcntl.h>
    #include <errno.h>

    using socket_t = int;
    #define INVALID_SOCKET (-1)
    #define SOCKET_ERROR   (-1)
    #define IS_INVALID_SOCKET(s) ((s) < 0)
    #define CLOSE_SOCKET(s) ::close(s)
    #define GET_SOCKET_ERR() (errno)

    inline void portable_sleep_micros(int64_t micros) {
        if (micros <= 0) return;
        ::usleep(static_cast<useconds_t>(micros));
    }

    inline void portable_sleep_seconds(int sec) {
        if (sec <= 0) return;
        ::sleep(static_cast<unsigned int>(sec));
    }

    inline void portable_yield() {
        ::usleep(1);
    }
#endif

// =============================================================================
// GLOBAL CONTROL & SIGNAL HANDLING
// =============================================================================
static std::atomic<bool> g_running{true};

void signal_handler(int sig) {
    (void)sig;
    g_running = false;
}

// =============================================================================
// ANSI TERMINAL FORMATTING CODES
// =============================================================================
namespace Color {
    const char* RESET       = "\033[0m";
    const char* BOLD        = "\033[1m";
    const char* DIM         = "\033[2m";
    const char* RED         = "\033[31m";
    const char* GREEN       = "\033[32m";
    const char* YELLOW      = "\033[33m";
    const char* BLUE        = "\033[34m";
    const char* MAGENTA     = "\033[35m";
    const char* CYAN        = "\033[36m";
    const char* WHITE       = "\033[37m";
    const char* ORANGE      = "\033[38;5;208m";
    const char* BG_DARK     = "\033[48;5;234m";
    const char* CLEAR_LINE  = "\033[2K\r";
}

// =============================================================================
// CONFIGURATION STRUCT
// =============================================================================
struct Config {
    std::string csv_path     = "";
    std::string host         = "127.0.0.1";
    int         port         = 8000;
    double      speed_mult   = 100.0;    // 100x acceleration by default
    bool        speed_max    = false;    // Max raw throughput mode (no sleep)
    double      fixed_delay  = -1.0;     // Fixed delay in ms (-1 means use timestamp delta)
    uint64_t    tx_limit     = 0;        // 0 = unlimited
    bool        loop_forever = false;    // Loop dataset endlessly
    bool        quiet        = false;    // Disable rich ANSI dashboard
};

// =============================================================================
// RUNTIME BENCHMARK METRICS
// =============================================================================
struct Metrics {
    std::atomic<uint64_t> total_sent{0};
    std::atomic<uint64_t> total_bytes{0};
    std::atomic<uint64_t> fraud_count{0};
    std::atomic<uint64_t> legit_count{0};
    
    // Latency tracking (in microseconds)
    std::atomic<uint64_t> latency_sum_us{0};
    std::atomic<uint64_t> min_latency_us{UINT64_MAX};
    std::atomic<uint64_t> max_latency_us{0};
    
    // Rolling TPS window
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point last_tick_time;
    uint64_t last_tick_sent = 0;
    double current_tps = 0.0;
    double avg_latency_us = 0.0;

    std::string last_tx_id       = "-";
    std::string last_pan         = "-";
    std::string last_amount      = "0.00";
    std::string last_category    = "-";
    int         last_is_fraud    = 0;
    std::string last_fraud_vec   = "Legitimate";
    std::string last_json_sample = "";
};

// =============================================================================
// SOCKET MANAGER CLASS (Thread-safe, resilient auto-reconnect)
// =============================================================================
class ResilientSocketClient {
public:
    ResilientSocketClient(const std::string& host, int port)
        : host_(host), port_(port), sock_(INVALID_SOCKET), is_connected_(false) {
        init_network_subsystem();
    }

    ~ResilientSocketClient() {
        disconnect();
        cleanup_network_subsystem();
    }

    bool connect_with_retry(int retry_delay_sec = 2) {
        while (g_running && !is_connected_) {
            sock_ = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
            if (IS_INVALID_SOCKET(sock_)) {
                std::cerr << Color::RED << "[!] Socket creation failed: " 
                          << GET_SOCKET_ERR() << Color::RESET << "\n";
                portable_sleep_seconds(retry_delay_sec);
                continue;
            }

            // Disable Nagle's algorithm for low-latency transmission
            int flag = 1;
            #ifdef _WIN32
                ::setsockopt(sock_, IPPROTO_TCP, TCP_NODELAY, (const char*)&flag, sizeof(flag));
            #else
                ::setsockopt(sock_, IPPROTO_TCP, TCP_NODELAY, (const void*)&flag, sizeof(flag));
            #endif

            // Set socket send buffer size (64 KB)
            int sndbuf = 65536;
            #ifdef _WIN32
                ::setsockopt(sock_, SOL_SOCKET, SO_SNDBUF, (const char*)&sndbuf, sizeof(sndbuf));
            #else
                ::setsockopt(sock_, SOL_SOCKET, SO_SNDBUF, (const void*)&sndbuf, sizeof(sndbuf));
            #endif

            struct sockaddr_in server_addr;
            std::memset(&server_addr, 0, sizeof(server_addr));
            server_addr.sin_family = AF_INET;
            server_addr.sin_port   = htons(static_cast<uint16_t>(port_));

            // Portable IP translation
            unsigned long in_addr = ::inet_addr(host_.c_str());
            if (in_addr != INADDR_NONE) {
                server_addr.sin_addr.s_addr = in_addr;
            } else {
                struct hostent* he = ::gethostbyname(host_.c_str());
                if (he == nullptr) {
                    std::cerr << Color::RED << "[!] Invalid address/hostname: " << host_ << Color::RESET << "\n";
                    CLOSE_SOCKET(sock_);
                    sock_ = INVALID_SOCKET;
                    portable_sleep_seconds(retry_delay_sec);
                    continue;
                }
                std::memcpy(&server_addr.sin_addr, he->h_addr_list[0], he->h_length);
            }

            // Attempt TCP Handshake
            if (::connect(sock_, (struct sockaddr*)&server_addr, sizeof(server_addr)) == 0) {
                is_connected_ = true;
                return true;
            }

            CLOSE_SOCKET(sock_);
            sock_ = INVALID_SOCKET;

            std::cout << Color::YELLOW << "[*] Waiting for AEGIS Blue Team Receiver on "
                      << host_ << ":" << port_ << "... (Retrying in " << retry_delay_sec << "s)"
                      << Color::RESET << "\r" << std::flush;

            portable_sleep_seconds(retry_delay_sec);
        }
        return is_connected_;
    }

    bool send_payload(const std::string& data, uint64_t& out_latency_us) {
        if (!is_connected_) {
            if (!connect_with_retry()) return false;
        }

        const char* buf = data.c_str();
        size_t total_bytes = data.size();
        size_t bytes_sent = 0;

        auto t0 = std::chrono::high_resolution_clock::now();

        while (bytes_sent < total_bytes && g_running) {
            #ifdef _WIN32
                int sent = ::send(sock_, buf + bytes_sent, static_cast<int>(total_bytes - bytes_sent), 0);
            #else
                ssize_t sent = ::send(sock_, buf + bytes_sent, total_bytes - bytes_sent, MSG_NOSIGNAL);
            #endif

            if (sent <= 0) {
                int err = GET_SOCKET_ERR();
                #ifdef _WIN32
                if (sent == SOCKET_ERROR && (err == WSAEWOULDBLOCK || err == WSAEINTR)) {
                #else
                if (sent < 0 && (err == EAGAIN || err == EWOULDBLOCK || err == EINTR)) {
                #endif
                    portable_yield();
                    continue;
                }
                
                // Connection broken
                disconnect();
                return false;
            }
            bytes_sent += sent;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        out_latency_us = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
        );

        return true;
    }

    void disconnect() {
        if (sock_ != INVALID_SOCKET) {
            CLOSE_SOCKET(sock_);
            sock_ = INVALID_SOCKET;
        }
        is_connected_ = false;
    }

    bool is_connected() const { return is_connected_; }

private:
    std::string host_;
    int port_;
    socket_t sock_;
    bool is_connected_;

    void init_network_subsystem() {
        #ifdef _WIN32
            WSADATA wsa;
            int res = WSAStartup(MAKEWORD(2, 2), &wsa);
            if (res != 0) {
                std::cerr << "[!] WSAStartup failed with error: " << res << "\n";
            }
        #endif
    }

    void cleanup_network_subsystem() {
        #ifdef _WIN32
            WSACleanup();
        #endif
    }
};

// =============================================================================
// CSV & STRING UTILITIES
// =============================================================================
class CSVHelper {
public:
    static std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(" \t\r\n\"");
        if (first == std::string::npos) return "";
        size_t last = str.find_last_not_of(" \t\r\n\"");
        return str.substr(first, (last - first + 1));
    }

    static std::string to_lower(std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return s;
    }

    // High-performance robust CSV tokenizer supporting quoted commas
    static void parse_csv_line(const std::string& line, std::vector<std::string>& out_fields) {
        out_fields.clear();
        std::string current;
        current.reserve(64);
        bool in_quotes = false;

        for (size_t i = 0; i < line.size(); ++i) {
            char c = line[i];
            if (c == '"') {
                if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                    current.push_back('"');
                    ++i; // skip escaped quote
                } else {
                    in_quotes = !in_quotes;
                }
            } else if (c == ',' && !in_quotes) {
                out_fields.push_back(trim(current));
                current.clear();
            } else if (c == '\r') {
                // Ignore CR in CRLF
                continue;
            } else {
                current.push_back(c);
            }
        }
        out_fields.push_back(trim(current));
    }

    // Escapes special characters for manual JSON serialization
    static std::string escape_json(const std::string& input) {
        std::string out;
        out.reserve(input.size() + 16);
        for (char c : input) {
            switch (c) {
                case '"':  out += "\\\""; break;
                case '\\': out += "\\\\"; break;
                case '\b': out += "\\b";  break;
                case '\f': out += "\\f";  break;
                case '\n': out += "\\n";  break;
                case '\r': out += "\\r";  break;
                case '\t': out += "\\t";  break;
                default:
                    if (static_cast<unsigned char>(c) < 0x20) {
                        char hex_buf[8];
                        std::snprintf(hex_buf, sizeof(hex_buf), "\\u%04x", c);
                        out += hex_buf;
                    } else {
                        out += c;
                    }
                    break;
            }
        }
        return out;
    }
};

// =============================================================================
// FAST TIME UTILITIES & UTC EPOCH CONVERTER (Zero OS dependency)
// =============================================================================
class TimeUtils {
public:
    // Pure arithmetic civil calendar to Unix epoch days (Howard Hinnant algorithm)
    static int64_t days_from_civil(int y, unsigned m, unsigned d) noexcept {
        y -= m <= 2;
        const int era = (y >= 0 ? y : y - 399) / 400;
        const unsigned yoe = static_cast<unsigned>(y - era * 400);      // [0, 399]
        const unsigned doy = (153 * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d - 1; // [0, 365]
        const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;    // [0, 146096]
        return era * 146097 + static_cast<int>(doe) - 719468;
    }

    // Fast ISO timestamp parser: YYYY-MM-DD HH:MM:SS.uuuuuu
    static int64_t parse_timestamp_to_micros(const std::string& ts_str) {
        if (ts_str.empty()) return 0;

        int year = 1970, month = 1, day = 1, hour = 0, min = 0, sec = 0;
        int micros = 0;

        char sep = ' ';
        int scanned = std::sscanf(ts_str.c_str(), "%d-%d-%d%c%d:%d:%d.%d",
                                  &year, &month, &day, &sep, &hour, &min, &sec, &micros);
        
        if (scanned < 7) {
            scanned = std::sscanf(ts_str.c_str(), "%d-%d-%d%c%d:%d:%d",
                                  &year, &month, &day, &sep, &hour, &min, &sec);
            if (scanned < 6) return 0;
            micros = 0;
        }

        int64_t days = days_from_civil(year, static_cast<unsigned>(month), static_cast<unsigned>(day));
        int64_t total_sec = days * 86400LL + hour * 3600LL + min * 60LL + sec;

        return (total_sec * 1000000LL) + micros;
    }
};

// =============================================================================
// DYNAMIC CSV SCHEMA MAPPER & ZERO-DEPENDENCY JSON SERIALIZER
// =============================================================================
class TransactionSchema {
public:
    std::vector<std::string> raw_headers;
    std::unordered_map<std::string, size_t> header_map;

    // Canonical column index bindings
    int idx_tx_id       = -1;
    int idx_timestamp   = -1;
    int idx_pan         = -1;
    int idx_merchant_id = -1;
    int idx_category    = -1;
    int idx_mcc         = -1;
    int idx_card_type   = -1;
    int idx_amount      = -1;
    
    // Behavioral Biometric Telemetry
    int idx_dwell_time  = -1;
    int idx_pressure    = -1;
    int idx_velocity    = -1;
    
    // Graph Centrality Features
    int idx_src_deg     = -1;
    int idx_dst_deg     = -1;
    int idx_deg_cent    = -1;
    int idx_src_pr      = -1;
    int idx_dst_pr      = -1;
    int idx_pagerank    = -1;
    int idx_src_close   = -1;
    int idx_dst_close   = -1;
    int idx_closeness   = -1;

    // NLP & Fraud labels
    int idx_text_memo   = -1;
    int idx_fraud_vec   = -1;
    int idx_is_fraud    = -1;

    bool initialize(const std::vector<std::string>& headers) {
        raw_headers = headers;
        header_map.clear();

        for (size_t i = 0; i < headers.size(); ++i) {
            std::string norm = CSVHelper::to_lower(headers[i]);
            header_map[norm] = i;
        }

        // Helper lambda for canonical mapping
        auto find_col = [&](const std::vector<std::string>& candidates) -> int {
            for (const auto& c : candidates) {
                auto it = header_map.find(CSVHelper::to_lower(c));
                if (it != header_map.end()) {
                    return static_cast<int>(it->second);
                }
            }
            return -1;
        };

        idx_tx_id       = find_col({"transactionid", "tx_id", "transaction_id", "id"});
        idx_timestamp   = find_col({"timestamp", "txtime", "time", "date"});
        idx_pan         = find_col({"pan", "card_id", "card_number", "account"});
        idx_merchant_id = find_col({"merchantid", "merchant_id", "merchid", "merchant"});
        idx_category    = find_col({"merchantcategory", "category", "merchant_category"});
        idx_mcc         = find_col({"mcc", "merchant_category_code"});
        idx_card_type   = find_col({"cardtype", "card_type", "type"});
        idx_amount      = find_col({"transactionamt", "amount", "transaction_amt", "amt"});

        idx_dwell_time  = find_col({"keystroke_dwell_time", "dwell_time", "keystroke_dwell"});
        idx_pressure    = find_col({"tap_pressure", "pressure", "touch_pressure"});
        idx_velocity    = find_col({"swipe_velocity", "velocity", "swipe_speed"});

        idx_src_deg     = find_col({"src_degree_centrality", "source_degree_centrality"});
        idx_dst_deg     = find_col({"dst_degree_centrality", "target_degree_centrality"});
        idx_deg_cent    = find_col({"degree_centrality", "deg_centrality"});
        
        idx_src_pr      = find_col({"src_pagerank", "source_pagerank"});
        idx_dst_pr      = find_col({"dst_pagerank", "target_pagerank"});
        idx_pagerank    = find_col({"pagerank", "page_rank"});

        idx_src_close   = find_col({"src_closeness_centrality", "source_closeness_centrality"});
        idx_dst_close   = find_col({"dst_closeness_centrality", "target_closeness_centrality"});
        idx_closeness   = find_col({"closeness_centrality", "closeness"});

        idx_text_memo   = find_col({"textmemo", "text_memo", "memo", "description"});
        idx_fraud_vec   = find_col({"fraudvector", "fraud_vector", "vector", "attack_type"});
        idx_is_fraud    = find_col({"isfraud", "is_fraud", "fraud", "label"});

        return true;
    }

    std::string get_field_or_default(const std::vector<std::string>& row, int idx, const std::string& def = "") const {
        if (idx >= 0 && idx < static_cast<int>(row.size())) {
            const std::string& val = row[idx];
            if (!val.empty()) return val;
        }
        return def;
    }

    double get_double_or_default(const std::vector<std::string>& row, int idx, double def = 0.0) const {
        if (idx >= 0 && idx < static_cast<int>(row.size())) {
            const std::string& val = row[idx];
            if (!val.empty()) {
                try {
                    return std::stod(val);
                } catch (...) {
                    return def;
                }
            }
        }
        return def;
    }

    int get_int_or_default(const std::vector<std::string>& row, int idx, int def = 0) const {
        if (idx >= 0 && idx < static_cast<int>(row.size())) {
            const std::string& val = row[idx];
            if (!val.empty()) {
                try {
                    return std::stoi(val);
                } catch (...) {
                    return def;
                }
            }
        }
        return def;
    }

    // High-performance direct JSON builder
    std::string to_json_payload(const std::vector<std::string>& row,
                                std::string& out_tx_id,
                                std::string& out_pan,
                                std::string& out_amount_str,
                                std::string& out_category,
                                int& out_is_fraud,
                                std::string& out_fraud_vec) const {
        
        out_tx_id      = get_field_or_default(row, idx_tx_id, "TX_00000000");
        std::string ts = get_field_or_default(row, idx_timestamp, "1970-01-01 00:00:00.000000");
        out_pan        = get_field_or_default(row, idx_pan, "CARD_UNKNOWN");
        std::string mid= get_field_or_default(row, idx_merchant_id, "MERCH_UNKNOWN");
        out_category   = get_field_or_default(row, idx_category, "Retail");
        std::string mcc= get_field_or_default(row, idx_mcc, "5411");
        std::string ctp= get_field_or_default(row, idx_card_type, "debit");
        
        double amt     = get_double_or_default(row, idx_amount, 0.0);
        double dwell   = get_double_or_default(row, idx_dwell_time, 80.0);
        double press   = get_double_or_default(row, idx_pressure, 0.35);
        double veloc   = get_double_or_default(row, idx_velocity, 2.0);

        double src_deg = get_double_or_default(row, idx_src_deg, 
                         get_double_or_default(row, idx_deg_cent, 0.0));
        double dst_deg = get_double_or_default(row, idx_dst_deg, 
                         get_double_or_default(row, idx_deg_cent, 0.0));

        double src_pr  = get_double_or_default(row, idx_src_pr, 
                         get_double_or_default(row, idx_pagerank, 0.0));
        double dst_pr  = get_double_or_default(row, idx_dst_pr, 
                         get_double_or_default(row, idx_pagerank, 0.0));

        double src_cls = get_double_or_default(row, idx_src_close, 
                         get_double_or_default(row, idx_closeness, 0.0));
        double dst_cls = get_double_or_default(row, idx_dst_close, 
                         get_double_or_default(row, idx_closeness, 0.0));

        std::string memo = get_field_or_default(row, idx_text_memo, "Payment Clearance");
        out_fraud_vec    = get_field_or_default(row, idx_fraud_vec, "Legitimate");
        out_is_fraud     = get_int_or_default(row, idx_is_fraud, 0);

        char amt_buf[32];
        std::snprintf(amt_buf, sizeof(amt_buf), "%.2f", amt);
        out_amount_str = amt_buf;

        // Manual JSON construction with reserve to minimize reallocations
        std::string json;
        json.reserve(512);

        json += "{";
        json += "\"TransactionID\":\"" + CSVHelper::escape_json(out_tx_id) + "\",";
        json += "\"Timestamp\":\"" + CSVHelper::escape_json(ts) + "\",";
        json += "\"PAN\":\"" + CSVHelper::escape_json(out_pan) + "\",";
        json += "\"MerchantID\":\"" + CSVHelper::escape_json(mid) + "\",";
        json += "\"MerchantCategory\":\"" + CSVHelper::escape_json(out_category) + "\",";
        json += "\"MCC\":\"" + CSVHelper::escape_json(mcc) + "\",";
        json += "\"CardType\":\"" + CSVHelper::escape_json(ctp) + "\",";
        json += "\"TransactionAmt\":" + std::to_string(amt) + ",";

        // Behavioral Biometrics
        json += "\"keystroke_dwell_time\":" + std::to_string(dwell) + ",";
        json += "\"tap_pressure\":" + std::to_string(press) + ",";
        json += "\"swipe_velocity\":" + std::to_string(veloc) + ",";

        // Graph Structural Features
        json += "\"src_degree_centrality\":" + std::to_string(src_deg) + ",";
        json += "\"dst_degree_centrality\":" + std::to_string(dst_deg) + ",";
        json += "\"degree_centrality\":" + std::to_string((src_deg + dst_deg) * 0.5) + ",";

        json += "\"src_pagerank\":" + std::to_string(src_pr) + ",";
        json += "\"dst_pagerank\":" + std::to_string(dst_pr) + ",";
        json += "\"pagerank\":" + std::to_string((src_pr + dst_pr) * 0.5) + ",";

        json += "\"src_closeness_centrality\":" + std::to_string(src_cls) + ",";
        json += "\"dst_closeness_centrality\":" + std::to_string(dst_cls) + ",";
        json += "\"closeness_centrality\":" + std::to_string((src_cls + dst_cls) * 0.5) + ",";

        // NLP & Ground Truth
        json += "\"TextMemo\":\"" + CSVHelper::escape_json(memo) + "\",";
        json += "\"FraudVector\":\"" + CSVHelper::escape_json(out_fraud_vec) + "\",";
        json += "\"IsFraud\":" + std::to_string(out_is_fraud);

        json += "}\n"; // Newline delimited JSON for TCP stream framing
        return json;
    }
};

// =============================================================================
// LIVE ANSI TERMINAL DASHBOARD
// =============================================================================
class Dashboard {
public:
    static void print_banner(const Config& cfg) {
        std::cout << "\033[2J\033[H"; // Clear screen & home cursor
        std::cout << Color::BOLD << Color::ORANGE
                  << "===============================================================================\n"
                  << "  PROJECT AEGIS : HIGH-SPEED PAYMENT ROUTER & ADVERSARIAL STREAM SIMULATOR      \n"
                  << "  Mastercard Innovation Challenge @ Global Fintech Fest 2026                   \n"
                  << "==============================================================================="
                  << Color::RESET << "\n";
        
        std::cout << Color::CYAN << "[SYSTEM CONFIG]" << Color::RESET << " "
                  << "Target: " << Color::BOLD << cfg.host << ":" << cfg.port << Color::RESET << " | "
                  << "Speed: " << Color::BOLD;
        if (cfg.speed_max) {
            std::cout << Color::RED << "MAX RAW THROUGHPUT (Stress-Test)" << Color::RESET;
        } else if (cfg.fixed_delay >= 0) {
            std::cout << cfg.fixed_delay << "ms fixed delay" << Color::RESET;
        } else {
            std::cout << cfg.speed_mult << "x accelerated" << Color::RESET;
        }
        std::cout << " | File: " << Color::DIM << cfg.csv_path << Color::RESET << "\n";
        std::cout << "-------------------------------------------------------------------------------\n";
    }

    static void render(const Config& cfg, Metrics& m, uint64_t total_csv_rows, bool is_connected) {
        if (cfg.quiet) return;

        auto now = std::chrono::steady_clock::now();
        double elapsed_sec = std::chrono::duration<double>(now - m.start_time).count();
        if (elapsed_sec <= 0.0001) elapsed_sec = 0.0001;

        // Calculate 1-second rolling TPS
        double tick_sec = std::chrono::duration<double>(now - m.last_tick_time).count();
        if (tick_sec >= 0.5) {
            uint64_t cur_sent = m.total_sent.load();
            m.current_tps = static_cast<double>(cur_sent - m.last_tick_sent) / tick_sec;
            m.last_tick_sent = cur_sent;
            m.last_tick_time = now;
            
            uint64_t sum_lat = m.latency_sum_us.load();
            if (cur_sent > 0) {
                m.avg_latency_us = static_cast<double>(sum_lat) / static_cast<double>(cur_sent);
            }
        }

        uint64_t sent = m.total_sent.load();
        uint64_t bytes = m.total_bytes.load();
        uint64_t fraud = m.fraud_count.load();
        uint64_t legit = m.legit_count.load();
        double fraud_pct = (sent > 0) ? (static_cast<double>(fraud) * 100.0 / sent) : 0.0;
        double lifetime_tps = static_cast<double>(sent) / elapsed_sec;

        // Move cursor to row 6 (right under header)
        std::cout << "\033[6;1H";

        // 1. Connection State & Status
        std::cout << Color::CLEAR_LINE;
        std::cout << Color::BOLD << "STATUS: " << Color::RESET;
        if (is_connected) {
            std::cout << Color::GREEN << "[ CONNECTED TO AEGIS BLUE TEAM RECEIVER ]" << Color::RESET;
        } else {
            std::cout << Color::RED << "[ RECONNECTING... ]" << Color::RESET;
        }
        std::cout << "   Elapsed: " << Color::CYAN << std::fixed << std::setprecision(1) 
                  << elapsed_sec << "s" << Color::RESET << "\n";

        // 2. Metrics Grid
        std::cout << Color::CLEAR_LINE;
        std::cout << "  • Streamed:  " << Color::BOLD << Color::WHITE << sent << Color::RESET;
        if (total_csv_rows > 0) {
            double progress = (static_cast<double>(sent % total_csv_rows) / total_csv_rows) * 100.0;
            std::cout << " / " << total_csv_rows << " (" << std::fixed << std::setprecision(1) << progress << "%)";
        }
        std::cout << "   • Volume: " << Color::BOLD << std::fixed << std::setprecision(2)
                  << (bytes / (1024.0 * 1024.0)) << " MB" << Color::RESET << "\n";

        // 3. Performance & Throughput
        std::cout << Color::CLEAR_LINE;
        std::cout << "  • Live TPS:  " << Color::BOLD << Color::GREEN << std::fixed << std::setprecision(1) 
                  << m.current_tps << " tx/s" << Color::RESET
                  << " (Avg: " << std::fixed << std::setprecision(1) << lifetime_tps << " tx/s)\n";

        // 4. Socket Latency
        std::cout << Color::CLEAR_LINE;
        uint64_t min_lat = m.min_latency_us.load();
        uint64_t max_lat = m.max_latency_us.load();
        if (min_lat == UINT64_MAX) min_lat = 0;

        std::cout << "  • Latency:   " << Color::BOLD << Color::CYAN 
                  << std::fixed << std::setprecision(2) << (m.avg_latency_us / 1000.0) << " ms" << Color::RESET
                  << " [Min: " << (min_lat / 1000.0) << " ms | Max: " << (max_lat / 1000.0) << " ms]\n";

        // 5. Fraud Distribution
        std::cout << Color::CLEAR_LINE;
        std::cout << "  • Payload:   Legit: " << Color::GREEN << legit << Color::RESET
                  << " | Fraud: " << Color::RED << fraud << Color::RESET
                  << " (" << Color::YELLOW << std::fixed << std::setprecision(2) << fraud_pct << "%" << Color::RESET << ")\n";

        // 6. Last Dispatched Transaction Card
        std::cout << Color::CLEAR_LINE << "-------------------------------------------------------------------------------\n";
        std::cout << Color::CLEAR_LINE;
        std::cout << Color::BOLD << "LAST DISPATCHED TRANSACTION:" << Color::RESET << "\n";
        std::cout << Color::CLEAR_LINE;
        std::cout << "  TX_ID: " << Color::YELLOW << m.last_tx_id << Color::RESET
                  << " | PAN: " << Color::CYAN << m.last_pan << Color::RESET
                  << " | Amt: " << Color::BOLD << "$" << m.last_amount << Color::RESET
                  << " | Cat: " << m.last_category << "\n";
        std::cout << Color::CLEAR_LINE;
        std::cout << "  Vector: ";
        if (m.last_is_fraud) {
            std::cout << Color::RED << Color::BOLD << "[FRAUD: " << m.last_fraud_vec << "]" << Color::RESET;
        } else {
            std::cout << Color::GREEN << "[LEGITIMATE]" << Color::RESET;
        }
        std::cout << "\n";
        std::cout << Color::CLEAR_LINE << "-------------------------------------------------------------------------------\n";
        std::cout << Color::DIM << "Press Ctrl+C to stop simulation and generate benchmark summary report." << Color::RESET << "\n";
        std::cout << std::flush;
    }

    static void print_final_summary(const Config& cfg, const Metrics& m) {
        auto now = std::chrono::steady_clock::now();
        double elapsed_sec = std::chrono::duration<double>(now - m.start_time).count();
        if (elapsed_sec <= 0.0001) elapsed_sec = 0.0001;

        uint64_t sent = m.total_sent.load();
        uint64_t bytes = m.total_bytes.load();
        uint64_t fraud = m.fraud_count.load();
        uint64_t legit = m.legit_count.load();
        double avg_tps = static_cast<double>(sent) / elapsed_sec;
        double avg_lat = (sent > 0) ? (static_cast<double>(m.latency_sum_us.load()) / sent) : 0.0;
        uint64_t min_lat = m.min_latency_us.load();
        uint64_t max_lat = m.max_latency_us.load();
        if (min_lat == UINT64_MAX) min_lat = 0;

        std::cout << "\n\n" << Color::BOLD << Color::ORANGE
                  << "===============================================================================\n"
                  << "  PROJECT AEGIS SIMULATOR : BENCHMARK & EXECUTION REPORT                       \n"
                  << "==============================================================================="
                  << Color::RESET << "\n";
        
        std::cout << "  Target Endpoint:           " << cfg.host << ":" << cfg.port << "\n";
        std::cout << "  Total Runtime:             " << std::fixed << std::setprecision(2) << elapsed_sec << " seconds\n";
        std::cout << "  Total Transactions Sent:   " << Color::BOLD << sent << Color::RESET << "\n";
        std::cout << "  Total Data Transmitted:    " << std::fixed << std::setprecision(2) << (bytes / (1024.0 * 1024.0)) << " MB\n";
        std::cout << "  Average Throughput:        " << Color::BOLD << Color::GREEN << std::fixed << std::setprecision(2) << avg_tps << " TPS (tx/sec)" << Color::RESET << "\n";
        std::cout << "  Mean Socket Write Latency: " << Color::CYAN << std::fixed << std::setprecision(2) << (avg_lat / 1000.0) << " ms (" << avg_lat << " µs)" << Color::RESET << "\n";
        std::cout << "  Min/Max Write Latency:     " << (min_lat / 1000.0) << " ms / " << (max_lat / 1000.0) << " ms\n";
        std::cout << "  Legitimate Transactions:   " << legit << "\n";
        std::cout << "  Fraud Attacks Injected:    " << Color::RED << fraud << Color::RESET 
                  << " (" << std::fixed << std::setprecision(2) << (sent > 0 ? (fraud * 100.0 / sent) : 0.0) << "%)\n";
        std::cout << "===============================================================================\n\n";
    }
};

// =============================================================================
// FILE DISCOVERY HELPER
// =============================================================================
std::string resolve_csv_path(const std::string& input_path) {
    if (!input_path.empty()) {
        std::ifstream test(input_path);
        if (test.good()) return input_path;
    }

    std::vector<std::string> search_candidates = {
        "aegis_synthetic_transactions.csv",
        "data/aegis_synthetic_transactions.csv",
        "../data/aegis_synthetic_transactions.csv",
        "../../data/aegis_synthetic_transactions.csv",
        "scratch/aegis_synthetic_transactions.csv",
        "../scratch/aegis_synthetic_transactions.csv"
    };

    for (const auto& path : search_candidates) {
        std::ifstream test(path);
        if (test.good()) {
            return path;
        }
    }
    return input_path; // Return original if nothing found
}

// =============================================================================
// CLI ARGUMENT PARSER
// =============================================================================
void print_help(const char* prog) {
    std::cout << Color::BOLD << "Project AEGIS Payment Gateway Simulator" << Color::RESET << "\n"
              << "Usage: " << prog << " [OPTIONS]\n\n"
              << "Options:\n"
              << "  -f, --file <path>       Path to synthetic CSV file (default: data/aegis_synthetic_transactions.csv)\n"
              << "  -h, --host <host>       Target receiver host/IP (default: 127.0.0.1)\n"
              << "  -p, --port <port>       Target receiver port (default: 8000)\n"
              << "  -s, --speed <val>       Speed multiplier: e.g., 1x (realtime), 10x, 100x, max (raw throughput)\n"
              << "  -d, --delay <ms>        Fixed millisecond delay between transactions (e.g., -d 5)\n"
              << "  -n, --limit <count>     Stop after streaming N transactions (0 = all)\n"
              << "  -l, --loop              Continuously loop dataset for endurance stress testing\n"
              << "  -q, --quiet             Disable live ANSI dashboard (log periodically)\n"
              << "  --help                  Show this help message and exit\n\n"
              << "Examples:\n"
              << "  " << prog << " --speed 100x\n"
              << "  " << prog << " --speed max --limit 50000\n"
              << "  " << prog << " --delay 10 --port 8000\n";
}

bool parse_args(int argc, char* argv[], Config& cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help") {
            print_help(argv[0]);
            return false;
        } else if ((arg == "-f" || arg == "--file") && i + 1 < argc) {
            cfg.csv_path = argv[++i];
        } else if ((arg == "-h" || arg == "--host") && i + 1 < argc) {
            cfg.host = argv[++i];
        } else if ((arg == "-p" || arg == "--port") && i + 1 < argc) {
            cfg.port = std::stoi(argv[++i]);
        } else if ((arg == "-s" || arg == "--speed") && i + 1 < argc) {
            std::string spd = argv[++i];
            std::string lower_spd = CSVHelper::to_lower(spd);
            if (lower_spd == "max" || lower_spd == "raw" || lower_spd == "0" || lower_spd == "inf") {
                cfg.speed_max = true;
            } else {
                if (spd.back() == 'x' || spd.back() == 'X') {
                    spd.pop_back();
                }
                cfg.speed_mult = std::stod(spd);
                if (cfg.speed_mult <= 0.0) cfg.speed_max = true;
            }
        } else if ((arg == "-d" || arg == "--delay") && i + 1 < argc) {
            cfg.fixed_delay = std::stod(argv[++i]);
        } else if ((arg == "-n" || arg == "--limit") && i + 1 < argc) {
            cfg.tx_limit = std::stoull(argv[++i]);
        } else if (arg == "-l" || arg == "--loop") {
            cfg.loop_forever = true;
        } else if (arg == "-q" || arg == "--quiet") {
            cfg.quiet = true;
        }
    }
    return true;
}

// =============================================================================
// MAIN STREAMING LOOP
// =============================================================================
int main(int argc, char* argv[]) {
    // Register signal handlers for clean SIGINT / SIGTERM teardown
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    Config cfg;
    if (!parse_args(argc, argv, cfg)) {
        return 0;
    }

    cfg.csv_path = resolve_csv_path(cfg.csv_path);

    std::ifstream csv_file(cfg.csv_path);
    if (!csv_file.is_open()) {
        std::cerr << Color::RED << "[!] Error: Unable to open CSV dataset at: " << cfg.csv_path << Color::RESET << "\n"
                  << "    Please verify file path with --file <path>.\n";
        return 1;
    }

    // 1. Read and parse dynamic header row
    std::string header_line;
    if (!std::getline(csv_file, header_line)) {
        std::cerr << Color::RED << "[!] Error: Empty CSV file.\n" << Color::RESET;
        return 1;
    }

    std::vector<std::string> headers;
    CSVHelper::parse_csv_line(header_line, headers);

    TransactionSchema schema;
    schema.initialize(headers);

    // Estimate row count for progress percentage
    uint64_t estimated_rows = 100000;

    // 2. Initialize Network Socket Client
    ResilientSocketClient client(cfg.host, cfg.port);
    
    if (!cfg.quiet) {
        Dashboard::print_banner(cfg);
    }
    std::cout << Color::CYAN << "[*] Initializing AEGIS High-Speed TCP Socket Client...\n" << Color::RESET;
    
    if (!client.connect_with_retry(2)) {
        std::cout << "\n[!] Shutdown requested before connection was established.\n";
        return 0;
    }

    Metrics metrics;
    metrics.start_time = std::chrono::steady_clock::now();
    metrics.last_tick_time = metrics.start_time;

    if (!cfg.quiet) {
        Dashboard::print_banner(cfg);
    }

    int64_t prev_sim_time_us = -1;
    auto last_dash_render = std::chrono::steady_clock::now();

    std::string line;
    std::vector<std::string> row_fields;
    row_fields.reserve(32);

    // 3. Execution Streaming Loop
    while (g_running) {
        if (!std::getline(csv_file, line)) {
            if (cfg.loop_forever && g_running) {
                csv_file.clear();
                csv_file.seekg(0, std::ios::beg);
                std::getline(csv_file, header_line); // Skip header on rewind
                prev_sim_time_us = -1;
                continue;
            } else {
                break; // End of file reached
            }
        }

        if (line.empty()) continue;

        CSVHelper::parse_csv_line(line, row_fields);
        if (row_fields.size() < 3) continue; // Skip malformed rows

        // Parse timestamp and compute pacing
        std::string ts_str = schema.get_field_or_default(row_fields, schema.idx_timestamp);
        int64_t cur_sim_time_us = TimeUtils::parse_timestamp_to_micros(ts_str);

        if (!cfg.speed_max) {
            if (cfg.fixed_delay >= 0.0) {
                // Fixed millisecond delay pacing
                if (cfg.fixed_delay > 0.0) {
                    portable_sleep_micros(static_cast<int64_t>(cfg.fixed_delay * 1000.0));
                }
            } else if (prev_sim_time_us > 0 && cur_sim_time_us >= prev_sim_time_us) {
                // Chronological time delta simulation
                int64_t delta_sim_us = cur_sim_time_us - prev_sim_time_us;
                int64_t target_sleep_us = static_cast<int64_t>(
                    static_cast<double>(delta_sim_us) / cfg.speed_mult
                );

                if (target_sleep_us > 0) {
                    // Cap max sleep to 5 seconds to prevent frozen stream on large CSV gaps
                    if (target_sleep_us > 5000000LL) target_sleep_us = 5000000LL;
                    portable_sleep_micros(target_sleep_us);
                }
            }
        }
        prev_sim_time_us = cur_sim_time_us;

        // Construct JSON Payload
        std::string tx_id, pan, amt_str, category, fraud_vec;
        int is_fraud = 0;

        std::string json_payload = schema.to_json_payload(
            row_fields, tx_id, pan, amt_str, category, is_fraud, fraud_vec
        );

        // Send over TCP Socket with latency profiling
        uint64_t latency_us = 0;
        bool sent_ok = client.send_payload(json_payload, latency_us);

        if (!sent_ok) {
            if (!g_running) break;
            // Retry loop kicks in
            if (!client.connect_with_retry(2)) break;
            // Resend current payload
            client.send_payload(json_payload, latency_us);
        }

        // Update Benchmark Metrics
        metrics.total_sent.fetch_add(1, std::memory_order_relaxed);
        metrics.total_bytes.fetch_add(json_payload.size(), std::memory_order_relaxed);
        metrics.latency_sum_us.fetch_add(latency_us, std::memory_order_relaxed);

        // Min/Max latency CAS loop
        uint64_t cur_min = metrics.min_latency_us.load(std::memory_order_relaxed);
        while (latency_us < cur_min && !metrics.min_latency_us.compare_exchange_weak(cur_min, latency_us)) {}
        
        uint64_t cur_max = metrics.max_latency_us.load(std::memory_order_relaxed);
        while (latency_us > cur_max && !metrics.max_latency_us.compare_exchange_weak(cur_max, latency_us)) {}

        if (is_fraud) {
            metrics.fraud_count.fetch_add(1, std::memory_order_relaxed);
        } else {
            metrics.legit_count.fetch_add(1, std::memory_order_relaxed);
        }

        metrics.last_tx_id     = tx_id;
        metrics.last_pan       = pan;
        metrics.last_amount    = amt_str;
        metrics.last_category  = category;
        metrics.last_is_fraud  = is_fraud;
        metrics.last_fraud_vec = fraud_vec;

        // Render Dashboard (every 100ms or every 100 txs)
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_dash_render).count() >= 100) {
            Dashboard::render(cfg, metrics, estimated_rows, client.is_connected());
            last_dash_render = now;
        }

        // Check transaction limit
        if (cfg.tx_limit > 0 && metrics.total_sent.load() >= cfg.tx_limit) {
            break;
        }
    }

    // Final render and clean disconnection
    Dashboard::render(cfg, metrics, estimated_rows, client.is_connected());
    client.disconnect();

    // Print final summary report
    Dashboard::print_final_summary(cfg, metrics);

    return 0;
}
