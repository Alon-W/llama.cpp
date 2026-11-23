set(TARGET llama-inkling-dspark-convert)
add_executable(${TARGET} inkling-dspark-convert.cpp)
target_link_libraries(${TARGET} PRIVATE ggml vendor::nlohmann)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
