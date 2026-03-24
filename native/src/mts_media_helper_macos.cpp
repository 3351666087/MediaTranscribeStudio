#include <algorithm>
#include <chrono>
#include <csignal>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

struct ProcessResult {
    int exit_code = -1;
    std::string stdout_text;
    std::string stderr_text;
    std::string error;
    bool timed_out = false;
};

struct ParsedArgs {
    std::unordered_map<std::string, std::string> values;
    std::vector<std::string> positionals;
};

struct CommandVariant {
    std::string label;
    std::vector<std::string> args;
};

std::string lower_copy(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return text;
}

std::string json_escape(const std::string& text) {
    std::ostringstream out;
    for (const unsigned char ch : text) {
        switch (ch) {
        case '\\':
            out << "\\\\";
            break;
        case '"':
            out << "\\\"";
            break;
        case '\b':
            out << "\\b";
            break;
        case '\f':
            out << "\\f";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            if (ch < 0x20U) {
                static const char* hex = "0123456789ABCDEF";
                out << "\\u00" << hex[(ch >> 4U) & 0x0FU] << hex[ch & 0x0FU];
            } else {
                out << static_cast<char>(ch);
            }
            break;
        }
    }
    return out.str();
}

ParsedArgs parse_args(int argc, char* argv[]) {
    ParsedArgs parsed;
    for (int i = 2; i < argc; ++i) {
        const std::string current = argv[i] ? std::string(argv[i]) : std::string();
        if (current.rfind("--", 0) == 0) {
            const std::string key = lower_copy(current.substr(2));
            std::string value = "1";
            if (i + 1 < argc) {
                const std::string next = argv[i + 1] ? std::string(argv[i + 1]) : std::string();
                if (next.rfind("--", 0) != 0) {
                    value = next;
                    ++i;
                }
            }
            parsed.values[key] = value;
            continue;
        }
        parsed.positionals.push_back(current);
    }
    return parsed;
}

std::optional<std::string> get_value(const ParsedArgs& parsed, const std::string& key) {
    const auto it = parsed.values.find(lower_copy(key));
    if (it == parsed.values.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::string require_value(const ParsedArgs& parsed, const std::string& key) {
    const auto value = get_value(parsed, key);
    if (!value || value->empty()) {
        throw std::runtime_error("missing required argument: " + key);
    }
    return *value;
}

int int_value(const ParsedArgs& parsed, const std::string& key, int default_value) {
    const auto value = get_value(parsed, key);
    if (!value || value->empty()) {
        return default_value;
    }
    try {
        return std::stoi(*value);
    } catch (...) {
        return default_value;
    }
}

bool bool_value(const ParsedArgs& parsed, const std::string& key, bool default_value) {
    const auto value = get_value(parsed, key);
    if (!value || value->empty()) {
        return default_value;
    }
    const std::string lowered = lower_copy(*value);
    return lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on";
}

void ensure_parent_dir(const fs::path& path) {
    std::error_code ec;
    const fs::path parent = path.parent_path();
    if (!parent.empty()) {
        fs::create_directories(parent, ec);
    }
}

bool file_exists_nonempty(const fs::path& path) {
    std::error_code ec;
    return fs::exists(path, ec) && fs::is_regular_file(path, ec) && fs::file_size(path, ec) > 0;
}

std::string ffmpeg_escape_filter_path(const fs::path& path) {
    std::string safe = path.lexically_normal().string();
    std::replace(safe.begin(), safe.end(), '\\', '/');
    std::string out;
    out.reserve(safe.size() + 8U);
    for (const char ch : safe) {
        if (ch == ':') {
            out += "\\:";
        } else if (ch == '\'') {
            out += "\\'";
        } else {
            out.push_back(ch);
        }
    }
    return out;
}

std::vector<std::string> build_video_encode_args(const ParsedArgs& parsed) {
    const std::string codec = require_value(parsed, "video-codec");
    const std::string preset = get_value(parsed, "preset").value_or("");
    const int ffmpeg_crf = std::max(0, int_value(parsed, "ffmpeg-crf", 20));
    const int webm_crf = std::max(0, std::min(63, int_value(parsed, "webm-crf", 32)));
    const int webm_cpu_used = std::max(0, int_value(parsed, "webm-cpu-used", 2));

    std::vector<std::string> out{"-pix_fmt", "yuv420p"};
    if (!codec.empty() && codec.size() >= 6U &&
        codec.compare(codec.size() - 6U, 6U, "_nvenc") == 0) {
        if (!preset.empty()) {
            out.insert(out.end(), {"-preset", preset});
        }
        out.insert(out.end(), {"-cq", std::to_string(ffmpeg_crf)});
        return out;
    }
    if (codec == "h264_videotoolbox" || codec == "hevc_videotoolbox") {
        out.insert(out.end(), {"-b:v", "0"});
        out.insert(out.end(), {"-q:v", codec == "h264_videotoolbox" ? "65" : "55"});
        return out;
    }
    if (codec == "libx264" || codec == "libx265") {
        if (!preset.empty()) {
            out.insert(out.end(), {"-preset", preset});
        }
        out.insert(out.end(), {"-crf", std::to_string(ffmpeg_crf)});
        return out;
    }
    if (codec == "mpeg4") {
        out.insert(out.end(), {"-q:v", "3"});
        return out;
    }
    if (codec == "libvpx-vp9") {
        out.insert(
            out.end(),
            {
                "-deadline",
                "good",
                "-cpu-used",
                std::to_string(webm_cpu_used),
                "-row-mt",
                "1",
                "-tile-columns",
                "1",
                "-crf",
                std::to_string(webm_crf),
                "-b:v",
                "0",
            }
        );
        return out;
    }
    if (codec == "libvpx") {
        out.insert(
            out.end(),
            {
                "-deadline",
                "good",
                "-cpu-used",
                std::to_string(webm_cpu_used),
                "-crf",
                std::to_string(std::max(4, webm_crf)),
                "-b:v",
                "0",
            }
        );
        return out;
    }
    return out;
}

bool is_videotoolbox_codec(const std::string& codec) {
    const std::string lowered = lower_copy(codec);
    return lowered == "h264_videotoolbox" || lowered == "hevc_videotoolbox";
}

int default_videotoolbox_bitrate_kbps(const ParsedArgs& parsed, bool is_hevc) {
    const int width = std::max(0, int_value(parsed, "width", 0));
    const int height = std::max(0, int_value(parsed, "height", 0));
    const long pixels = static_cast<long>(width) * static_cast<long>(height);
    if (pixels >= 2560L * 1440L) {
        return is_hevc ? 9000 : 14000;
    }
    if (pixels >= 1920L * 1080L) {
        return is_hevc ? 5000 : 8000;
    }
    if (pixels >= 1280L * 720L) {
        return is_hevc ? 3000 : 5000;
    }
    return is_hevc ? 1800 : 2800;
}

std::string kbps_text(int kbps) {
    return std::to_string(std::max(250, kbps)) + "k";
}

std::vector<CommandVariant> build_video_encode_variants(const ParsedArgs& parsed) {
    const std::string codec = require_value(parsed, "video-codec");
    if (!is_videotoolbox_codec(codec)) {
        return {
            {
                "default",
                build_video_encode_args(parsed),
            },
        };
    }

    const bool is_hevc = lower_copy(codec) == "hevc_videotoolbox";
    const std::string qv = is_hevc ? "55" : "65";
    const int bitrate_kbps = default_videotoolbox_bitrate_kbps(parsed, is_hevc);
    const std::string bitrate = kbps_text(bitrate_kbps);
    const std::string maxrate = kbps_text(static_cast<int>(bitrate_kbps * 1.35));
    const std::string bufsize = kbps_text(static_cast<int>(bitrate_kbps * 2.0));

    return {
        {
            "vt-apple-speed",
            {
                "-pix_fmt",
                "nv12",
                "-allow_sw",
                "1",
                "-realtime",
                "true",
                "-prio_speed",
                "true",
                "-b:v",
                bitrate,
                "-maxrate",
                maxrate,
                "-bufsize",
                bufsize,
                "-g",
                "60",
            },
        },
        {
            "vt-apple-balanced",
            {
                "-pix_fmt",
                "yuv420p",
                "-allow_sw",
                "1",
                "-realtime",
                "true",
                "-b:v",
                bitrate,
                "-maxrate",
                maxrate,
                "-bufsize",
                bufsize,
                "-g",
                "60",
            },
        },
        {
            "vt-qv",
            {
                "-pix_fmt",
                "yuv420p",
                "-b:v",
                "0",
                "-q:v",
                qv,
            },
        },
        {
            "vt-allow-sw",
            {"-pix_fmt", "yuv420p", "-allow_sw", "1", "-b:v", bitrate},
        },
        {
            "vt-nv12",
            {
                "-pix_fmt",
                "nv12",
                "-allow_sw",
                "1",
                "-b:v",
                bitrate,
                "-maxrate",
                maxrate,
                "-bufsize",
                bufsize,
            },
        },
    };
}

std::vector<CommandVariant> build_input_variants(const ParsedArgs& parsed) {
    const std::string codec = require_value(parsed, "video-codec");
    if (!is_videotoolbox_codec(codec)) {
        return {
            {
                "default",
                {},
            },
        };
    }

    return {
        {
            "hwdec",
            {
                "-hwaccel",
                "videotoolbox",
            },
        },
        {
            "default",
            {},
        },
    };
}

std::vector<std::string> build_audio_encode_args(const ParsedArgs& parsed) {
    const std::string codec = get_value(parsed, "audio-codec").value_or("");
    const std::string webm_audio_bitrate =
        get_value(parsed, "webm-audio-bitrate").value_or("160k");
    if (codec.empty() || codec == "copy") {
        return {};
    }
    if (codec == "libopus" || codec == "opus") {
        return {"-b:a", webm_audio_bitrate, "-vbr", "on"};
    }
    if (codec == "libvorbis") {
        return {"-q:a", "5"};
    }
    if (codec == "aac") {
        return {"-b:a", "192k"};
    }
    return {};
}

std::string build_subtitles_filter(const ParsedArgs& parsed) {
    const fs::path subtitle_path = fs::path(require_value(parsed, "subtitle-path"));
    std::ostringstream filter;
    filter << "subtitles='" << ffmpeg_escape_filter_path(subtitle_path) << "':charenc=UTF-8";

    const int width = std::max(0, int_value(parsed, "width", 0));
    const int height = std::max(0, int_value(parsed, "height", 0));
    if (width > 0 && height > 0) {
        filter << ":original_size=" << width << "x" << height;
    }

    const std::string fonts_dir = get_value(parsed, "fonts-dir").value_or("");
    if (!fonts_dir.empty()) {
        const fs::path font_path = fs::path(fonts_dir);
        if (fs::exists(font_path) && fs::is_directory(font_path)) {
            filter << ":fontsdir='" << ffmpeg_escape_filter_path(font_path) << "'";
        }
    }

    const std::string force_style = get_value(parsed, "force-style").value_or("");
    if (!force_style.empty()) {
        std::string fixed;
        fixed.reserve(force_style.size() + 4U);
        for (const char ch : force_style) {
            if (ch == '\'') {
                fixed += "\\'";
            } else {
                fixed.push_back(ch);
            }
        }
        filter << ":force_style='" << fixed << "'";
    }
    return filter.str();
}

std::string build_ass_filter(const ParsedArgs& parsed) {
    const fs::path subtitle_path = fs::path(require_value(parsed, "subtitle-path"));
    std::ostringstream filter;
    filter << "ass='" << ffmpeg_escape_filter_path(subtitle_path) << "'";

    const int width = std::max(0, int_value(parsed, "width", 0));
    const int height = std::max(0, int_value(parsed, "height", 0));
    if (width > 0 && height > 0) {
        filter << ":original_size=" << width << "x" << height;
    }

    const std::string fonts_dir = get_value(parsed, "fonts-dir").value_or("");
    if (!fonts_dir.empty()) {
        const fs::path font_path = fs::path(fonts_dir);
        if (fs::exists(font_path) && fs::is_directory(font_path)) {
            filter << ":fontsdir='" << ffmpeg_escape_filter_path(font_path) << "'";
        }
    }
    filter << ":shaping=auto";
    return filter.str();
}

std::string build_filter(const ParsedArgs& parsed) {
    const std::string kind = lower_copy(require_value(parsed, "filter-kind"));
    if (kind == "subtitles") {
        return build_subtitles_filter(parsed);
    }
    if (kind == "ass") {
        return build_ass_filter(parsed);
    }
    throw std::runtime_error("unsupported filter kind: " + kind);
}

void read_fd_thread(int fd, std::string* out) {
    char buffer[4096];
    while (true) {
        const ssize_t read_bytes = ::read(fd, buffer, sizeof(buffer));
        if (read_bytes > 0) {
            out->append(buffer, static_cast<std::size_t>(read_bytes));
            continue;
        }
        if (read_bytes == 0) {
            break;
        }
        if (errno == EINTR) {
            continue;
        }
        break;
    }
    ::close(fd);
}

ProcessResult run_process(const std::vector<std::string>& args, unsigned long timeout_ms) {
    ProcessResult result;
    if (args.empty()) {
        result.error = "empty command";
        return result;
    }

    int stdout_pipe[2] = {-1, -1};
    int stderr_pipe[2] = {-1, -1};
    if (::pipe(stdout_pipe) != 0) {
        result.error = "pipe(stdout) failed";
        return result;
    }
    if (::pipe(stderr_pipe) != 0) {
        result.error = "pipe(stderr) failed";
        ::close(stdout_pipe[0]);
        ::close(stdout_pipe[1]);
        return result;
    }

    const pid_t pid = ::fork();
    if (pid < 0) {
        result.error = "fork failed";
        ::close(stdout_pipe[0]);
        ::close(stdout_pipe[1]);
        ::close(stderr_pipe[0]);
        ::close(stderr_pipe[1]);
        return result;
    }

    if (pid == 0) {
        ::dup2(stdout_pipe[1], STDOUT_FILENO);
        ::dup2(stderr_pipe[1], STDERR_FILENO);
        ::close(stdout_pipe[0]);
        ::close(stdout_pipe[1]);
        ::close(stderr_pipe[0]);
        ::close(stderr_pipe[1]);

        std::vector<char*> argv_ptrs;
        argv_ptrs.reserve(args.size() + 1U);
        for (const auto& arg : args) {
            argv_ptrs.push_back(const_cast<char*>(arg.c_str()));
        }
        argv_ptrs.push_back(nullptr);

        ::execvp(argv_ptrs[0], argv_ptrs.data());
        const std::string msg = "execvp failed: " + std::string(std::strerror(errno));
        ::write(STDERR_FILENO, msg.c_str(), msg.size());
        ::write(STDERR_FILENO, "\n", 1);
        _exit(127);
    }

    ::close(stdout_pipe[1]);
    ::close(stderr_pipe[1]);

    std::thread stdout_thread(read_fd_thread, stdout_pipe[0], &result.stdout_text);
    std::thread stderr_thread(read_fd_thread, stderr_pipe[0], &result.stderr_text);

    int status = 0;
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (true) {
        const pid_t wait_rc = ::waitpid(pid, &status, WNOHANG);
        if (wait_rc == pid) {
            break;
        }
        if (wait_rc < 0) {
            result.error = "waitpid failed";
            break;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            result.timed_out = true;
            result.error = "process timed out";
            ::kill(pid, SIGTERM);
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            if (::waitpid(pid, &status, WNOHANG) == 0) {
                ::kill(pid, SIGKILL);
            }
            ::waitpid(pid, &status, 0);
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    if (WIFEXITED(status)) {
        result.exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        result.exit_code = 128 + WTERMSIG(status);
    }

    stdout_thread.join();
    stderr_thread.join();
    return result;
}

std::string best_error_line(const std::string& text) {
    std::vector<std::string> lines;
    std::istringstream stream(text);
    std::string line;
    while (std::getline(stream, line)) {
        if (!line.empty()) {
            lines.push_back(line);
        }
    }
    if (lines.empty()) {
        return {};
    }

    const std::vector<std::string> preferred_markers = {
        "cannot create compression session",
        "hardware encoder may be busy",
        "error while opening encoder",
        "-12908",
        "videotoolbox",
    };
    for (const auto& marker : preferred_markers) {
        for (const auto& raw : lines) {
            const std::string lowered = lower_copy(raw);
            if (lowered.find(marker) != std::string::npos) {
                return raw;
            }
        }
    }

    for (auto it = lines.rbegin(); it != lines.rend(); ++it) {
        const std::string lowered = lower_copy(*it);
        if (lowered.find("nothing was written into output file") != std::string::npos) {
            continue;
        }
        return *it;
    }
    return lines.back();
}

int print_json_error(const std::string& message) {
    std::cerr << "{\"error\":\"" << json_escape(message) << "\"}" << std::endl;
    return 1;
}

int handle_probe(const ParsedArgs& parsed) {
    const std::string ffprobe = require_value(parsed, "ffprobe");
    const std::string input = require_value(parsed, "input");
    const unsigned long timeout_ms =
        static_cast<unsigned long>(std::max(1, int_value(parsed, "timeout", 30)) * 1000);

    const ProcessResult result = run_process(
        {ffprobe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", input},
        timeout_ms
    );
    if (!result.error.empty() && result.timed_out) {
        return print_json_error(result.error);
    }
    if (result.exit_code != 0) {
        const std::string tail =
            best_error_line(result.stderr_text.empty() ? result.stdout_text : result.stderr_text);
        return print_json_error(tail.empty() ? "ffprobe failed" : tail);
    }
    std::cout << result.stdout_text;
    return 0;
}

int handle_encoders(const ParsedArgs& parsed) {
    const std::string ffmpeg = require_value(parsed, "ffmpeg");
    const unsigned long timeout_ms =
        static_cast<unsigned long>(std::max(1, int_value(parsed, "timeout", 30)) * 1000);

    const ProcessResult result = run_process({ffmpeg, "-hide_banner", "-encoders"}, timeout_ms);
    if (!result.error.empty() && result.timed_out) {
        return print_json_error(result.error);
    }
    if (result.exit_code != 0) {
        const std::string tail =
            best_error_line(result.stderr_text.empty() ? result.stdout_text : result.stderr_text);
        return print_json_error(tail.empty() ? "ffmpeg -encoders failed" : tail);
    }
    std::cout << result.stdout_text;
    if (!result.stderr_text.empty()) {
        if (!result.stdout_text.empty() &&
            result.stdout_text.back() != '\n' &&
            result.stdout_text.back() != '\r') {
            std::cout << '\n';
        }
        std::cout << result.stderr_text;
    }
    return 0;
}

int handle_burn(const ParsedArgs& parsed) {
    const std::string ffmpeg = require_value(parsed, "ffmpeg");
    const std::string input = require_value(parsed, "input");
    const std::string output = require_value(parsed, "output");
    const std::string video_codec = require_value(parsed, "video-codec");
    const std::string filter = build_filter(parsed);
    const std::string audio_codec = get_value(parsed, "audio-codec").value_or("");
    const unsigned long timeout_ms =
        static_cast<unsigned long>(std::max(1, int_value(parsed, "timeout", 7200)) * 1000);

    ensure_parent_dir(fs::path(output));

    const std::vector<std::string> base_cmd{
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
    };

    const auto audio_args = build_audio_encode_args(parsed);
    const auto input_variants = build_input_variants(parsed);
    const auto video_variants = build_video_encode_variants(parsed);
    std::vector<std::string> errors;

    for (const auto& input_variant : input_variants) {
        for (const auto& variant : video_variants) {
            std::error_code remove_ec;
            fs::remove(fs::path(output), remove_ec);

            std::vector<std::string> cmd = base_cmd;
            cmd.insert(cmd.end(), input_variant.args.begin(), input_variant.args.end());
            cmd.insert(
                cmd.end(),
                {
                    "-i",
                    input,
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a?",
                    "-map_metadata",
                    "0",
                    "-map_chapters",
                    "0",
                    "-sn",
                    "-dn",
                    "-vf",
                    filter,
                    "-c:v",
                    video_codec,
                }
            );
            cmd.insert(cmd.end(), variant.args.begin(), variant.args.end());

            if (!audio_codec.empty()) {
                cmd.insert(cmd.end(), {"-c:a", audio_codec});
                cmd.insert(cmd.end(), audio_args.begin(), audio_args.end());
            } else {
                cmd.push_back("-an");
            }

            if (bool_value(parsed, "movflags-faststart", false)) {
                cmd.insert(cmd.end(), {"-movflags", "+faststart"});
            }
            cmd.push_back(output);

            const ProcessResult result = run_process(cmd, timeout_ms);
            if (!result.error.empty() && result.timed_out) {
                return print_json_error(
                    input_variant.label + "/" + variant.label + ":" + result.error
                );
            }
            if (result.exit_code == 0 && file_exists_nonempty(fs::path(output))) {
                std::cout
                    << "{\"ok\":true,\"input_variant\":\""
                    << json_escape(input_variant.label)
                    << "\",\"video_variant\":\""
                    << json_escape(variant.label)
                    << "\"}"
                    << std::endl;
                return 0;
            }
            std::string tail =
                best_error_line(result.stderr_text.empty() ? result.stdout_text : result.stderr_text);
            if (tail.empty()) {
                tail = "ffmpeg burn failed";
            }
            errors.push_back(input_variant.label + "/" + variant.label + ":" + tail);
        }
    }

    if (errors.empty()) {
        return print_json_error("ffmpeg burn failed");
    }

    std::ostringstream summary;
    const std::size_t start =
        errors.size() > 5U ? (errors.size() - 5U) : 0U;
    for (std::size_t i = start; i < errors.size(); ++i) {
        if (i > start) {
            summary << " | ";
        }
        summary << errors[i];
    }
    return print_json_error(summary.str());
}

int handle_mux(const ParsedArgs& parsed) {
    const std::string ffmpeg = require_value(parsed, "ffmpeg");
    const std::string input_video = require_value(parsed, "input-video");
    const std::string subtitle_input = require_value(parsed, "subtitle-input");
    const std::string subtitle_codec = require_value(parsed, "subtitle-codec");
    const std::string output = require_value(parsed, "output");
    const unsigned long timeout_ms =
        static_cast<unsigned long>(std::max(1, int_value(parsed, "timeout", 7200)) * 1000);

    ensure_parent_dir(fs::path(output));

    const ProcessResult result = run_process(
        {
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            input_video,
            "-i",
            subtitle_input,
            "-map",
            "0",
            "-map",
            "1:0",
            "-c",
            "copy",
            "-c:s",
            subtitle_codec,
            "-metadata:s:s:0",
            "title=Transcription",
            "-disposition:s:0",
            "default",
            output,
        },
        timeout_ms
    );

    if (!result.error.empty() && result.timed_out) {
        return print_json_error(result.error);
    }
    if (result.exit_code != 0) {
        const std::string tail =
            best_error_line(result.stderr_text.empty() ? result.stdout_text : result.stderr_text);
        return print_json_error(tail.empty() ? "ffmpeg subtitle mux failed" : tail);
    }
    if (!file_exists_nonempty(fs::path(output))) {
        return print_json_error("subtitle mux finished but output video was not generated");
    }
    return 0;
}

void print_usage() {
    std::cerr
        << "mts_media_helper <probe|encoders|burn|mux> [options]\n"
        << "  probe    --ffprobe PATH --input PATH [--timeout SEC]\n"
        << "  encoders --ffmpeg PATH [--timeout SEC]\n"
        << "  burn     --ffmpeg PATH --input PATH --output PATH --subtitle-path PATH\n"
        << "           --filter-kind subtitles|ass --video-codec CODEC [--audio-codec CODEC]\n"
        << "  mux      --ffmpeg PATH --input-video PATH --subtitle-input PATH\n"
        << "           --subtitle-codec CODEC --output PATH\n";
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 2;
    }

    try {
        const std::string command = lower_copy(argv[1] ? std::string(argv[1]) : std::string());
        const ParsedArgs parsed = parse_args(argc, argv);

        if (command == "probe") {
            return handle_probe(parsed);
        }
        if (command == "encoders") {
            return handle_encoders(parsed);
        }
        if (command == "burn") {
            return handle_burn(parsed);
        }
        if (command == "mux") {
            return handle_mux(parsed);
        }
        print_usage();
        return 2;
    } catch (const std::exception& exc) {
        std::cerr << "{\"error\":\"" << json_escape(exc.what()) << "\"}" << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "{\"error\":\"unknown helper failure\"}" << std::endl;
        return 1;
    }
}
