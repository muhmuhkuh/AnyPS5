#include "GraphicsTests.hpp"
#include <cstdio>
#include <cstdlib>
#include <exception>
#ifdef _WIN32
#include <stdlib.h>
#endif

int main() {
#ifdef _WIN32
    _putenv_s("APS5_PIN_WAIT_MS", "200");
    _putenv_s("APS5_HEAP_MIRROR_MIB", "4");
#else
    setenv("APS5_PIN_WAIT_MS", "200", 1);
    setenv("APS5_HEAP_MIRROR_MIB", "4", 1);
#endif
    try { RunGuestTextureResourceTests(); return 0; }
    catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
