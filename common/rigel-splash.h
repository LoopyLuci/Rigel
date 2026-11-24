// Rigel terminal splash (Windows and AMD GPUs).
// Shown only on an interactive terminal; a program that captures the output (ModelMistress) never sees it.
#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#if defined(_WIN32)
#include <io.h>
#define isatty _isatty
#define fileno _fileno
#else
#include <unistd.h>
#endif

inline void rigel_show_splash(bool use_color = true) {
    if (!isatty(fileno(stdout))) {
        return;
    }

    static bool shown = false;
    if (shown) {
        return;
    }
    shown = true;

    const char * no_color = std::getenv("NO_COLOR");
    const char * term = std::getenv("TERM");
    if (no_color && *no_color) {
        use_color = false;
    }
    if (term && std::strcmp(term, "dumb") == 0) {
        use_color = false;
    }

    const char * txt = "RIGEL";
    const size_t len = std::strlen(txt);
    for (size_t i = 1; i <= len; ++i) {
        std::string s;
        for (size_t j = 0; j < i; ++j) {
            if (j > 0) s += " ";
            s += txt[j];
        }
        if (use_color) {
            std::fprintf(stdout, "\r\033[1;38;2;140;180;255m  %s\033[0m", s.c_str());
        } else {
            std::fprintf(stdout, "\r  %s", s.c_str());
        }
        std::fflush(stdout);
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
    }
    if (use_color) {
        std::fprintf(stdout, "\033[38;2;200;215;255m  \xc2\xb7 BETA-ORIONIS\033[0m\n");
        std::fprintf(stdout, "  \033[38;2;160;160;175mdemand-paged MoE \xc2\xb7 Windows + AMD HIP\033[0m\n\n");
    } else {
        std::fprintf(stdout, "  - BETA-ORIONIS\n  demand-paged MoE - Windows + AMD HIP\n\n");
    }
    std::fflush(stdout);
}
