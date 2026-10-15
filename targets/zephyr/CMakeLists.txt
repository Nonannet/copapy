cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(coparun)

# Core of the runner, shared with the other runners
set(CORE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../src/coparun)

target_sources(app PRIVATE
    src/main.c
    src/tasks.c
    ${CORE_DIR}/runmem.c
    ${CORE_DIR}/mem_man.c
)
target_include_directories(app PRIVATE ${CORE_DIR})

if(CONFIG_COPARUN_LOGGING)
    target_compile_definitions(app PRIVATE ENABLE_LOGGING)
else()
    target_compile_definitions(app PRIVATE ENABLE_BASIC_LOGGING)
endif()
