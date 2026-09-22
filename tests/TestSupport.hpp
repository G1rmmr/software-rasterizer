#pragma once
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace test {
    inline void Check(bool condition, const char* expression, const char* file, int line) {
        if(condition) return;
        std::ostringstream message;
        message << file << ':' << line << ": " << expression;
        throw std::runtime_error(message.str());
    }
    inline void Near(double actual, double expected, double tolerance, const char* file, int line) {
        if(std::isfinite(actual) && std::isfinite(expected) && std::abs(actual - expected) <= tolerance) return;
        std::ostringstream message;
        message << file << ':' << line << ": expected " << expected << ", got " << actual << " (tolerance " << tolerance
                << ')';
        throw std::runtime_error(message.str());
    }
}
#define CHECK(...) ::test::Check(static_cast<bool>((__VA_ARGS__)), #__VA_ARGS__, __FILE__, __LINE__)
#define CHECK_NEAR(actual, expected, tolerance) ::test::Near((actual), (expected), (tolerance), __FILE__, __LINE__)
#define CHECK_THROWS(...)                                                                                              \
    do {                                                                                                               \
        bool testDidThrow = false;                                                                                     \
        try {                                                                                                          \
            static_cast<void>((__VA_ARGS__));                                                                          \
        }                                                                                                              \
        catch(const std::exception&) {                                                                                 \
            testDidThrow = true;                                                                                       \
        }                                                                                                              \
        CHECK(testDidThrow);                                                                                           \
    } while(false)
