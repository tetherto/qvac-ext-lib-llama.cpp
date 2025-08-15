#pragma once

#include <cstdint>
#include <cstring>
#include <iostream>
#include <streambuf>
#include <vector>

/// @brief Custom traits for uint8_t for usage in std template classes that use char_traits (e.g. std::basic_streambuf)
template <> struct std::char_traits<uint8_t> {
    using char_type  = uint8_t;
    using int_type   = int;
    using off_type   = std::streamoff;
    using pos_type   = std::streampos;
    using state_type = std::mbstate_t;

    static void assign(char_type & c1, const char_type & c2) noexcept { c1 = c2; }

    static constexpr bool eq(char_type a, char_type b) noexcept { return a == b; }

    static constexpr bool lt(char_type a, char_type b) noexcept { return a < b; }

    static int compare(const char_type * s1, const char_type * s2, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            if (lt(s1[i], s2[i])) {
                return -1;
            }
            if (lt(s2[i], s1[i])) {
                return 1;
            }
        }
        return 0;
    }

    static std::size_t length(const char_type * s) {
        std::size_t i = 0;
        while (!eq(s[i], char_type())) {
            ++i;
        }
        return i;
    }

    static const char_type * find(const char_type * s, std::size_t n, const char_type & c) {
        for (std::size_t i = 0; i < n; ++i) {
            if (eq(s[i], c)) {
                return s + i;
            }
        }
        return nullptr;
    }

    static char_type * move(char_type * s1, const char_type * s2, std::size_t n) {
        return static_cast<char_type *>(std::memmove(s1, s2, n));
    }

    static char_type * copy(char_type * s1, const char_type * s2, std::size_t n) {
        return static_cast<char_type *>(std::memcpy(s1, s2, n));
    }

    static char_type * assign(char_type * s, std::size_t n, char_type c) {
        for (std::size_t i = 0; i < n; ++i) {
            s[i] = c;
        }
        return s;
    }

    static constexpr int_type not_eof(int_type c) noexcept { return eq_int_type(c, eof()) ? 0 : c; }

    static constexpr char_type to_char_type(int_type c) noexcept {
        return c >= 0 && c <= 255 ? static_cast<char_type>(c) : char_type();
    }

    static constexpr int_type to_int_type(char_type c) noexcept { return static_cast<int_type>(c); }

    static constexpr bool eq_int_type(int_type c1, int_type c2) noexcept { return c1 == c2; }

    static constexpr int_type eof() noexcept { return static_cast<int_type>(-1); }
};

/// @brief Custom streambuf for uint8_t
class Uint8BufferStreamBuf : public std::basic_streambuf<uint8_t> {
  public:
    Uint8BufferStreamBuf(std::vector<uint8_t> && _data);

  protected:
    int_type underflow() override;

    /// @brief Efficient bulk reading. The standard implementation specifies that this function can be overridden
    /// to provide a more efficient implementation: sgetn will call this function if it is overridden.
    std::streamsize xsgetn(char_type * s, std::streamsize n) override;

    pos_type seekoff(off_type off, std::ios_base::seekdir dir,
                     std::ios_base::openmode which = std::ios_base::in) override;

    pos_type seekpos(pos_type pos, std::ios_base::openmode which = std::ios_base::in) override;

  private:
    std::vector<uint8_t> data;
};
