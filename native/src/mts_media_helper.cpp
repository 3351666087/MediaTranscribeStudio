#include <algorithm>
#include <chrono>
#include <cwctype>
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

#ifdef _WIN32
#include <windows.h>
#endif

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
    std::unordered_map<std::wstring, std::wstring> values;
    std::vector<std::wstring> positionals;
};

std::wstring lower_copy(std::wstring text) {
    std::transform(text.begin(), text.end(), text.begin(), [](wchar_t ch) {
        return static_cast<wchar_t>(std::towlower(ch));
    });
    return text;
}

std::string utf8_from_wide(const std::wstring& text) {
#ifdef _WIN32
    if (text.empty()) {
        return {};
    }
    const int required = WideCharToMultiByte(
        CP_UTF8, 0, text.c_str(), static_cast<int>(text.size()), nullptr, 0, nullptr, nullptr
    );
    if (required <= 0) {
        return {};
    }
    std::string out(static_cast<std::size_t>(required), '\0');
    WideCharToMultiByte(
        CP_UTF8,
        0,
        text.c_str(),
        static_cast<int>(text.size()),
        out.data(),
        required,
        nullptr,
        nullptr
    );
    return out;
#else
    return std::string(text.begin(), text.end());
#endif
}

std::wstring wide_from_utf8(const std::string& text) {
#ifdef _WIN32
    if (text.empty()) {
        return {};
    }
    const int required = MultiByteToWideChar(
        CP_UTF8, 0, text.c_str(), static_cast<int>(text.size()), nullptr, 0
    );
    if (required <= 0) {
        return {};
    }
    std::wstring out(static_cast<std::size_t>(required), L'\0');
    MultiByteToWideChar(
        CP_UTF8, 0, text.c_str(), static_cast<int>(text.size()), out.data(), required
    );
    return out;
#else
    return std::wstring(text.begin(), text.end());
#endif
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

std::wstring quote_arg(const std::wstring& arg) {
    if (arg.empty()) {
        return L"\"\"";
    }
    if (arg.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
        return arg;
    }

    std::wstring out;
    out.push_back(L'"');
    std::size_t backslashes = 0;
    for (const wchar_t ch : arg) {
        if (ch == L'\\') {
            ++backslashes;
            continue;
        }
        if (ch == L'"') {
            out.append(backslashes * 2U + 1U, L'\\');
            out.push_back(L'"');
            backslashes = 0;
            continue;
        }
        if (backslashes > 0) {
            out.append(backslashes, L'\\');
            backslashes = 0;
        }
        out.push_back(ch);
    }
    if (backslashes > 0) {
        out.append(backslashes * 2U, L'\\');
    }
    out.push_back(L'"');
    return out;
}

std::wstring build_command_line(const std::vector<std::wstring>& args) {
    std::wstring command_line;
    bool first = true;
    for (const auto& arg : args) {
        if (!first) {
            command_line.push_back(L' ');
        }
        first = false;
        command_line += quote_arg(arg);
    }
    return command_line;
}

ParsedArgs parse_args(int argc, wchar_t* argv[]) {
    ParsedArgs parsed;
    for (int i = 2; i < argc; ++i) {
        std::wstring current = argv[i] ? std::wstring(argv[i]) : std::wstring();
        if (current.rfind(L"--", 0) == 0) {
            std::wstring key = lower_copy(current.substr(2));
            std::wstring value = L"1";
            if (i + 1 < argc) {
                std::wstring next = argv[i + 1] ? std::wstring(argv[i + 1]) : std::wstring();
                if (next.rfind(L"--", 0) != 0) {
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

std::optional<std::wstring> get_value(const ParsedArgs& parsed, const std::wstring& key) {
    const auto it = parsed.values.find(lower_copy(key));
    if (it == parsed.values.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::wstring require_value(const ParsedArgs& parsed, const std::wstring& key) {
    const auto value = get_value(parsed, key);
    if (!value || value->empty()) {
        throw std::runtime_error("missing required argument: " + utf8_from_wide(key));
    }
    return *value;
}

int int_value(const ParsedArgs& parsed, const std::wstring& key, int default_value) {
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

bool bool_value(const ParsedArgs& parsed, const std::wstring& key, bool default_value) {
    const auto value = get_value(parsed, key);
    if (!value || value->empty()) {
        return default_value;
    }
    const std::wstring lowered = lower_copy(*value);
    return lowered == L"1" || lowered == L"true" || lowered == L"yes" || lowered == L"on";
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

std::wstring ffmpeg_escape_filter_path(const fs::path& path) {
    std::wstring safe = path.lexically_normal().wstring();
    std::replace(safe.begin(), safe.end(), L'\\', L'/');
    std::wstring out;
    out.reserve(safe.size() + 8U);
    for (const wchar_t ch : safe) {
        if (ch == L':') {
            out += L"\\:";
        } else if (ch == L'\'') {
            out += L"\\'";
        } else {
            out.push_back(ch);
        }
    }
    return out;
}

std::vector<std::wstring> build_video_encode_args(const ParsedArgs& parsed) {
    const std::wstring codec = require_value(parsed, L"video-codec");
    const std::wstring preset = get_value(parsed, L"preset").value_or(L"");
    const int ffmpeg_crf = std::max(0, int_value(parsed, L"ffmpeg-crf", 20));
    const int webm_crf = std::max(0, std::min(63, int_value(parsed, L"webm-crf", 32)));
    const int webm_cpu_used = std::max(0, int_value(parsed, L"webm-cpu-used", 2));

    std::vector<std::wstring> out{L"-pix_fmt", L"yuv420p"};
    if (!codec.empty() && codec.size() >= 6U &&
        codec.compare(codec.size() - 6U, 6U, L"_nvenc") == 0) {
        if (!preset.empty()) {
            out.insert(out.end(), {L"-preset", preset});
        }
        out.insert(out.end(), {L"-cq", std::to_wstring(ffmpeg_crf)});
        return out;
    }
    if (codec == L"libx264" || codec == L"libx265") {
        if (!preset.empty()) {
            out.insert(out.end(), {L"-preset", preset});
        }
        out.insert(out.end(), {L"-crf", std::to_wstring(ffmpeg_crf)});
        return out;
    }
    if (codec == L"mpeg4") {
        out.insert(out.end(), {L"-q:v", L"3"});
        return out;
    }
    if (codec == L"libvpx-vp9") {
        out.insert(
            out.end(),
            {L"-deadline",
             L"good",
             L"-cpu-used",
             std::to_wstring(webm_cpu_used),
             L"-row-mt",
             L"1",
             L"-tile-columns",
             L"1",
             L"-crf",
             std::to_wstring(webm_crf),
             L"-b:v",
             L"0"}
        );
        return out;
    }
    if (codec == L"libvpx") {
        out.insert(
            out.end(),
            {L"-deadline",
             L"good",
             L"-cpu-used",
             std::to_wstring(webm_cpu_used),
             L"-crf",
             std::to_wstring(std::max(4, webm_crf)),
             L"-b:v",
             L"0"}
        );
        return out;
    }
    return out;
}

std::vector<std::wstring> build_audio_encode_args(const ParsedArgs& parsed) {
    const std::wstring codec = get_value(parsed, L"audio-codec").value_or(L"");
    const std::wstring webm_audio_bitrate =
        get_value(parsed, L"webm-audio-bitrate").value_or(L"160k");
    if (codec.empty() || codec == L"copy") {
        return {};
    }
    if (codec == L"libopus" || codec == L"opus") {
        return {L"-b:a", webm_audio_bitrate, L"-vbr", L"on"};
    }
    if (codec == L"libvorbis") {
        return {L"-q:a", L"5"};
    }
    if (codec == L"aac") {
        return {L"-b:a", L"192k"};
    }
    return {};
}

std::wstring build_subtitles_filter(const ParsedArgs& parsed) {
    const fs::path subtitle_path = fs::path(require_value(parsed, L"subtitle-path"));
    std::wostringstream filter;
    filter << L"subtitles='" << ffmpeg_escape_filter_path(subtitle_path) << L"':charenc=UTF-8";

    const int width = std::max(0, int_value(parsed, L"width", 0));
    const int height = std::max(0, int_value(parsed, L"height", 0));
    if (width > 0 && height > 0) {
        filter << L":original_size=" << width << L"x" << height;
    }

    const std::wstring fonts_dir = get_value(parsed, L"fonts-dir").value_or(L"");
    if (!fonts_dir.empty()) {
        const fs::path font_path = fs::path(fonts_dir);
        if (fs::exists(font_path) && fs::is_directory(font_path)) {
            filter << L":fontsdir='" << ffmpeg_escape_filter_path(font_path) << L"'";
        }
    }

    const std::wstring force_style = get_value(parsed, L"force-style").value_or(L"");
    if (!force_style.empty()) {
        std::wstring escaped = force_style;
        std::wstring fixed;
        fixed.reserve(escaped.size() + 4U);
        for (const wchar_t ch : escaped) {
            if (ch == L'\'') {
                fixed += L"\\'";
            } else {
                fixed.push_back(ch);
            }
        }
        filter << L":force_style='" << fixed << L"'";
    }
    return filter.str();
}

std::wstring build_ass_filter(const ParsedArgs& parsed) {
    const fs::path subtitle_path = fs::path(require_value(parsed, L"subtitle-path"));
    std::wostringstream filter;
    filter << L"ass='" << ffmpeg_escape_filter_path(subtitle_path) << L"'";

    const int width = std::max(0, int_value(parsed, L"width", 0));
    const int height = std::max(0, int_value(parsed, L"height", 0));
    if (width > 0 && height > 0) {
        filter << L":original_size=" << width << L"x" << height;
    }

    const std::wstring fonts_dir = get_value(parsed, L"fonts-dir").value_or(L"");
    if (!fonts_dir.empty()) {
        const fs::path font_path = fs::path(fonts_dir);
        if (fs::exists(font_path) && fs::is_directory(font_path)) {
            filter << L":fontsdir='" << ffmpeg_escape_filter_path(font_path) << L"'";
        }
    }
    filter << L":shaping=auto";
    return filter.str();
}

std::wstring build_filter(const ParsedArgs& parsed) {
    const std::wstring kind = lower_copy(require_value(parsed, L"filter-kind"));
    if (kind == L"subtitles") {
        return build_subtitles_filter(parsed);
    }
    if (kind == L"ass") {
        return build_ass_filter(parsed);
    }
    throw std::runtime_error("unsupported filter kind: " + utf8_from_wide(kind));
}

#ifdef _WIN32
void read_pipe_thread(HANDLE pipe_handle, std::string* out) {
    char buffer[4096];
    DWORD read_bytes = 0;
    while (ReadFile(pipe_handle, buffer, static_cast<DWORD>(sizeof(buffer)), &read_bytes, nullptr) &&
           read_bytes > 0) {
        out->append(buffer, buffer + read_bytes);
    }
    CloseHandle(pipe_handle);
}

ProcessResult run_process(const std::vector<std::wstring>& args, DWORD timeout_ms) {
    ProcessResult result;
    if (args.empty()) {
        result.error = "empty command";
        return result;
    }

    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    sa.lpSecurityDescriptor = nullptr;

    HANDLE stdout_read = nullptr;
    HANDLE stdout_write = nullptr;
    HANDLE stderr_read = nullptr;
    HANDLE stderr_write = nullptr;

    if (!CreatePipe(&stdout_read, &stdout_write, &sa, 0)) {
        result.error = "CreatePipe(stdout) failed";
        return result;
    }
    if (!SetHandleInformation(stdout_read, HANDLE_FLAG_INHERIT, 0)) {
        result.error = "SetHandleInformation(stdout) failed";
        CloseHandle(stdout_read);
        CloseHandle(stdout_write);
        return result;
    }
    if (!CreatePipe(&stderr_read, &stderr_write, &sa, 0)) {
        result.error = "CreatePipe(stderr) failed";
        CloseHandle(stdout_read);
        CloseHandle(stdout_write);
        return result;
    }
    if (!SetHandleInformation(stderr_read, HANDLE_FLAG_INHERIT, 0)) {
        result.error = "SetHandleInformation(stderr) failed";
        CloseHandle(stdout_read);
        CloseHandle(stdout_write);
        CloseHandle(stderr_read);
        CloseHandle(stderr_write);
        return result;
    }

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESHOWWINDOW | STARTF_USESTDHANDLES;
    startup.wShowWindow = SW_HIDE;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    startup.hStdOutput = stdout_write;
    startup.hStdError = stderr_write;

    PROCESS_INFORMATION process{};
    std::wstring command_line = build_command_line(args);
    std::vector<wchar_t> buffer(command_line.begin(), command_line.end());
    buffer.push_back(L'\0');

    const BOOL ok = CreateProcessW(
        nullptr,
        buffer.data(),
        nullptr,
        nullptr,
        TRUE,
        CREATE_NO_WINDOW,
        nullptr,
        nullptr,
        &startup,
        &process
    );

    CloseHandle(stdout_write);
    CloseHandle(stderr_write);

    if (!ok) {
        result.error = "CreateProcessW failed";
        CloseHandle(stdout_read);
        CloseHandle(stderr_read);
        return result;
    }

    std::thread stdout_thread(read_pipe_thread, stdout_read, &result.stdout_text);
    std::thread stderr_thread(read_pipe_thread, stderr_read, &result.stderr_text);

    const DWORD wait_rc = WaitForSingleObject(process.hProcess, timeout_ms);
    if (wait_rc == WAIT_TIMEOUT) {
        result.timed_out = true;
        result.error = "process timed out";
        TerminateProcess(process.hProcess, 124);
        WaitForSingleObject(process.hProcess, 5000);
    }

    DWORD exit_code = 0;
    if (GetExitCodeProcess(process.hProcess, &exit_code)) {
        result.exit_code = static_cast<int>(exit_code);
    }

    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);

    stdout_thread.join();
    stderr_thread.join();
    return result;
}
#else
ProcessResult run_process(const std::vector<std::wstring>&, unsigned long) {
    ProcessResult result;
    result.error = "unsupported platform";
    return result;
}
#endif

std::string last_nonempty_line(const std::string& text) {
    std::istringstream stream(text);
    std::string line;
    std::string last;
    while (std::getline(stream, line)) {
        if (!line.empty()) {
            last = line;
        }
    }
    return last;
}

int print_json_error(const std::string& message) {
    std::cerr << "{\"error\":\"" << json_escape(message) << "\"}" << std::endl;
    return 1;
}

int handle_probe(const ParsedArgs& parsed) {
    const std::wstring ffprobe = require_value(parsed, L"ffprobe");
    const std::wstring input = require_value(parsed, L"input");
    const unsigned long timeout_ms =
        static_cast<unsigned long>(std::max(1, int_value(parsed, L"timeout", 30)) * 1000);

    const ProcessResult result = run_process(
        {ffprobe,
         L"-v",
         L"error",
         L"-print_format",
         L"json",
         L"-show_streams",
         L"-show_format",
         input},
        timeout_ms
    );
    if (!result.error.empty() && result.timed_out) {
        return print_json_error(result.error);
    }
    if (result.exit_code != 0) {
        const std::string tail =
            last_nonempty_line(result.stderr_text.empty() ? result.stdout_text : result.stderr_text);
        return print_json_error(tail.empty() ? "ffprobe failed" : tail);
    }
    std::cout << result.stdout_text;
    return 0;
}

int handle_encoders(const ParsedArgs& parsed) {
    const std::wstring ffmpeg = require_value(parsed, L"ffmpeg");
    const unsigned long timeout_ms =
        static_cast<unsigned long>(std::max(1, int_value(parsed, L"timeout", 30)) * 1000);

    const ProcessResult result =
        run_process({ffmpeg, L"-hide_banner", L"-encoders"}, timeout_ms);
    if (!result.error.empty() && result.timed_out) {
        return print_json_error(result.error);
    }
    if (result.exit_code != 0) {
        const std::string tail =
            last_nonempty_line(result.stderr_text.empty() ? result.stdout_text : result.stderr_text);
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
    const std::wstring ffmpeg = require_value(parsed, L"ffmpeg");
    const std::wstring input = require_value(parsed, L"input");
    const std::wstring output = require_value(parsed, L"output");
    const std::wstring video_codec = require_value(parsed, L"video-codec");
    const std::wstring filter = build_filter(parsed);
    const std::wstring audio_codec = get_value(parsed, L"audio-codec").value_or(L"");
    const unsigned long timeout_ms =
        static_cast<unsigned long>(std::max(1, int_value(parsed, L"timeout", 7200)) * 1000);

    ensure_parent_dir(fs::path(output));

    std::vector<std::wstring> cmd{
        ffmpeg,         L"-y",         L"-hide_banner", L"-loglevel",   L"error",
        L"-i",          input,         L"-map",         L"0:v:0",       L"-map",
        L"0:a?",        L"-map_metadata", L"0",         L"-map_chapters", L"0",
        L"-sn",         L"-dn",        L"-vf",          filter,         L"-c:v",
        video_codec};

    const auto video_args = build_video_encode_args(parsed);
    cmd.insert(cmd.end(), video_args.begin(), video_args.end());

    if (!audio_codec.empty()) {
        cmd.insert(cmd.end(), {L"-c:a", audio_codec});
        const auto audio_args = build_audio_encode_args(parsed);
        cmd.insert(cmd.end(), audio_args.begin(), audio_args.end());
    } else {
        cmd.push_back(L"-an");
    }

    if (bool_value(parsed, L"movflags-faststart", false)) {
        cmd.insert(cmd.end(), {L"-movflags", L"+faststart"});
    }
    cmd.push_back(output);

    const ProcessResult result = run_process(cmd, timeout_ms);
    if (!result.error.empty() && result.timed_out) {
        return print_json_error(result.error);
    }
    if (result.exit_code != 0) {
        const std::string tail =
            last_nonempty_line(result.stderr_text.empty() ? result.stdout_text : result.stderr_text);
        return print_json_error(tail.empty() ? "ffmpeg burn failed" : tail);
    }
    if (!file_exists_nonempty(fs::path(output))) {
        return print_json_error("ffmpeg burn finished but output video was not generated");
    }
    return 0;
}

int handle_mux(const ParsedArgs& parsed) {
    const std::wstring ffmpeg = require_value(parsed, L"ffmpeg");
    const std::wstring input_video = require_value(parsed, L"input-video");
    const std::wstring subtitle_input = require_value(parsed, L"subtitle-input");
    const std::wstring subtitle_codec = require_value(parsed, L"subtitle-codec");
    const std::wstring output = require_value(parsed, L"output");
    const unsigned long timeout_ms =
        static_cast<unsigned long>(std::max(1, int_value(parsed, L"timeout", 7200)) * 1000);

    ensure_parent_dir(fs::path(output));

    const ProcessResult result = run_process(
        {ffmpeg,
         L"-y",
         L"-hide_banner",
         L"-loglevel",
         L"error",
         L"-i",
         input_video,
         L"-i",
         subtitle_input,
         L"-map",
         L"0",
         L"-map",
         L"1:0",
         L"-c",
         L"copy",
         L"-c:s",
         subtitle_codec,
         L"-metadata:s:s:0",
         L"title=Transcription",
         L"-disposition:s:0",
         L"default",
         output},
        timeout_ms
    );

    if (!result.error.empty() && result.timed_out) {
        return print_json_error(result.error);
    }
    if (result.exit_code != 0) {
        const std::string tail =
            last_nonempty_line(result.stderr_text.empty() ? result.stdout_text : result.stderr_text);
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

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
    if (argc < 2) {
        print_usage();
        return 2;
    }

#ifndef _WIN32
    std::vector<std::wstring> wide_args;
    wide_args.reserve(static_cast<std::size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        wide_args.push_back(wide_from_utf8(argv[i] ? std::string(argv[i]) : std::string()));
    }
    std::vector<wchar_t*> wide_ptrs;
    wide_ptrs.reserve(wide_args.size());
    for (auto& item : wide_args) {
        wide_ptrs.push_back(item.data());
    }
    argv = reinterpret_cast<char**>(wide_ptrs.data());
#endif

    try {
#ifdef _WIN32
        const std::wstring command = lower_copy(argv[1] ? std::wstring(argv[1]) : std::wstring());
        const ParsedArgs parsed = parse_args(argc, argv);
#else
        const std::wstring command =
            lower_copy(wide_ptrs.size() > 1 ? std::wstring(wide_ptrs[1]) : std::wstring());
        const ParsedArgs parsed = parse_args(argc, wide_ptrs.data());
#endif

        if (command == L"probe") {
            return handle_probe(parsed);
        }
        if (command == L"encoders") {
            return handle_encoders(parsed);
        }
        if (command == L"burn") {
            return handle_burn(parsed);
        }
        if (command == L"mux") {
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
