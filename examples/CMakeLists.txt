set(TARGET pyannote-diarization)
add_executable(${TARGET} main.cpp common_audio.cpp vbx.cpp)
target_link_libraries(${TARGET} PRIVATE ggml)
install(TARGETS ${TARGET} RUNTIME)

set(TARGET_STREAM pyannote-diarization-streaming)
add_executable(${TARGET_STREAM} main_streaming.cpp common_audio.cpp)
target_link_libraries(${TARGET_STREAM} PRIVATE ggml)
install(TARGETS ${TARGET_STREAM} RUNTIME)