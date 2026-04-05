set(TARGET pyannote-diarization)
add_executable(${TARGET}
    main.cpp
    common_audio.cpp
    pyannote_diarization.cpp
)
target_link_libraries(${TARGET} PRIVATE pyannote ggml)
install(TARGETS ${TARGET} RUNTIME)
