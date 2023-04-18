project(COCO
        DESCRIPTION "COCO code-experiment library"
        LANGUAGES C)
remove_definitions(-std=c++17)
find_library(MATH_LIBRARY m)
message("Math library: " ${MATH_LIBRARY})

include(CheckCompilerFlag)
check_compiler_flag(C "-pedantic -Wall -Wextra -Wstrict-prototypes -Wshadow -Wno-sign-compare -Wconversion" CC_HAS_WALL_ETC)

add_library(coco STATIC coco.c coco.h)
target_include_directories(coco PUBLIC .)
if(MATH_LIBRARY)
    target_link_libraries(coco PUBLIC ${MATH_LIBRARY})
endif()
# Add warning flags
if (MSVC)
  target_compile_options(coco PRIVATE "/W3")
elseif (CC_HAS_WALL_ETC)
  target_compile_options(coco PRIVATE -pedantic -Wall -Wextra -Wstrict-prototypes -Wshadow -Wno-sign-compare -Wconversion)
endif()

