set(PYANNOTE_SOURCES
    diarization.cpp
    ../../examples/vbx.cpp
)

add_library(pyannote ${PYANNOTE_SOURCES})
add_library(pyannote::pyannote ALIAS pyannote)

target_include_directories(pyannote
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../..
        ${CMAKE_CURRENT_SOURCE_DIR}/../../examples
        ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/kaldi-native-fbank
)

target_link_libraries(pyannote PUBLIC ggml kaldi-native-fbank-core)

target_compile_features(pyannote PUBLIC cxx_std_17)
