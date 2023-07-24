# list(APPEND CMAKE_PREFIX_PATH "/home/hiba/Desktop/PFE/ns-allinone-3.38/ns-3.38/contrib/libtorch")
# # set(Torch_DIR "/home/hiba/Desktop/PFE/ns-allinone-3.38/ns-3.38/contrib/libtorch/share/cmake/Torch")
# find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# if(${Torch_FOUND})
#   include_directories("/home/hiba/Desktop/PFE/ns-allinone-3.38/ns-3.38/contrib/libtorch/include")
#   set(torch_libraries ${Torch_LIBRARIES})
# endif()


# build_example(
#   NAME FLnode
#   HEADER_FILE_ONLY model/FL-node.h
#   SOURCE_FILES model/FL-node.cc
#   LIBRARIES_TO_LINK ${libinternet}
#                     ${libnetwork}
#                     ${libapplications}
#                     ${Torch_LIBRARIES}
# )

# add_executable(FLnode model/FL-node.h)
# target_link_libraries(FLnode "${TORCH_LIBRARIES}")
# set_property(TARGET FLnode PROPERTY CXX_STANDARD 17)

# find_external_library(DEPENDENCY_NAME Torch
#                       HEADER_NAME
#                         torch/script.h
#                         torch.h
#                       LIBRARY_NAME torch
#                       SEARCH_PATHS /home/hiba/Desktop/PFE/ns-allinone-3.38/ns-3.38/contrib/libtorch)



# option(DOWNLOAD_MNIST "Download the MNIST dataset from the internet" ON)
# if (DOWNLOAD_MNIST)
#   message(STATUS "Downloading MNIST dataset")
#   execute_process(
#     COMMAND python ${CMAKE_CURRENT_LIST_DIR}/../tools/download_mnist.py
#       -d ${CMAKE_BINARY_DIR}/data
#     ERROR_VARIABLE DOWNLOAD_ERROR)
#   if (DOWNLOAD_ERROR)
#     message(FATAL_ERROR "Error downloading MNIST dataset: ${DOWNLOAD_ERROR}")
#   endif()
# endif()

# add_custom_target(FLnode model/FL-node.h)
# target_compile_features(FLnode PUBLIC cxx_range_for)
# target_link_libraries(FLnode ${TORCH_LIBRARIES})

build_lib(
  LIBNAME applications
  SOURCE_FILES
    helper/bulk-send-helper.cc
    helper/on-off-helper.cc
    helper/packet-sink-helper.cc
    helper/three-gpp-http-helper.cc
    helper/udp-client-server-helper.cc
    helper/udp-echo-helper.cc
    helper/bitcoin-miner-helper.cc
    helper/bitcoin-node-helper.cc
    helper/bitcoin-topology-helper.cc
    model/application-packet-probe.cc
    model/bulk-send-application.cc
    model/onoff-application.cc
    model/packet-loss-counter.cc
    model/packet-sink.cc
    model/seq-ts-echo-header.cc
    model/seq-ts-header.cc
    model/seq-ts-size-header.cc
    model/three-gpp-http-client.cc
    model/three-gpp-http-header.cc
    model/three-gpp-http-server.cc
    model/three-gpp-http-variables.cc
    model/udp-client.cc
    model/udp-echo-client.cc
    model/udp-echo-server.cc
    model/udp-server.cc
    model/udp-trace-client.cc
    model/bitcoin-miner.cc
    model/bitcoin-node.cc
    model/bitcoin-selfish-miner-trials.cc
    model/bitcoin-selfish-miner.cc
    model/bitcoin-simple-attacker.cc
    model/bitcoin.cc
    model/FL-task-initiator.cc
    model/FL-node.cc
  HEADER_FILES
    helper/bulk-send-helper.h
    helper/on-off-helper.h
    helper/packet-sink-helper.h
    helper/three-gpp-http-helper.h
    helper/udp-client-server-helper.h
    helper/udp-echo-helper.h
    helper/bitcoin-miner-helper.h
    helper/bitcoin-node-helper.h
    helper/bitcoin-topology-helper.h
    model/application-packet-probe.h
    model/bulk-send-application.h
    model/onoff-application.h
    model/packet-loss-counter.h
    model/packet-sink.h
    model/seq-ts-echo-header.h
    model/seq-ts-header.h
    model/seq-ts-size-header.h
    model/three-gpp-http-client.h
    model/three-gpp-http-header.h
    model/three-gpp-http-server.h
    model/three-gpp-http-variables.h
    model/udp-client.h
    model/udp-echo-client.h
    model/udp-echo-server.h
    model/udp-server.h
    model/udp-trace-client.h
    model/bitcoin-miner.h
    model/bitcoin-node.h
    model/bitcoin-selfish-miner-trials.h
    model/bitcoin-selfish-miner.h
    model/bitcoin-simple-attacker.h
    model/bitcoin.h
    model/FL-task-initiator.h
    model/FL-node.h
  LIBRARIES_TO_LINK ${libinternet}
                    ${libstats}
                    # ${TORCH_LIBRARIES}
  TEST_SOURCES
    test/three-gpp-http-client-server-test.cc
    test/bulk-send-application-test-suite.cc
    test/udp-client-server-test.cc
)




# set(Torch_DIR "/home/hiba/Desktop/PFE/ns-allinone-3.38/ns-3.38/contrib/libtorch/share/cmake/Torch")
# find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# target_link_libraries(applications "${TORCH_LIBRARIES}")
# set_property(TARGET applications PROPERTY CXX_STANDARD 17)