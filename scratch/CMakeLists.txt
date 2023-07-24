set(target_prefix scratch_)
# # #torch
# find_external_library(DEPENDENCY_NAME Torch
#                       HEADER_NAME
#                         torch/script.h
#                         torch/torch.h
#                       LIBRARY_NAME torch
#                       SEARCH_PATHS $ENV{LIBTORCHPATH})

# if(${Torch_FOUND})
#   include_directories(${Torch_INCLUDE_DIRS})
#   link_libraries(${Torch_LIBRARIES})
# endif()

# set(torch_libraries)
# if(${Torch_FOUND})
#   set(torch_libraries ${Torch_LIBRARIES})
# endif()

# set(CMAKE_PREFIX_PATH "/home/hiba/Desktop/PFE/ns-allinone-3.38/ns-3.38/contrib/libtorch/share/cmake/Torch")


# find_package(Torch REQUIRED) 
# if(${Torch_FOUND}) 
# message("Torch found !!")
# # include_directories(${Torch_INCLUDE_DIRS})
# add_executable(simulator scratch-simulator.cc)
# target_link_libraries(simulator "${TORCH_LIBRARIES}")
# target_include_directories(simulator PRIVATE "/home/hiba/Desktop/PFE/ns-allinone-3.38/ns-3.38/contrib/libtorch/include/torch")
# set_property(TARGET simulator PROPERTY CXX_STANDARD 14)
# endif()

# include(FindPkgConfig)
# if(PKG_CONFIG_FOUND)
# message("PKG found!!!")

# pkg_check_modules(EXTERNAL_LIBRARY Torch PATH "/home/hiba/Desktop/PFE/ns-allinone-3.38/libtorch")
# endif()

# if(PKG_CONFIG_FOUND AND EXTERNAL_LIBRARY_FOUND)
# include_directories(${TORCH_INCLUDE_DIRS})
# set(external_libraries ${TORCH_LIBRARIES})
# endif()

function(create_scratch source_files)
  # Return early if no sources in the subdirectory
  list(LENGTH source_files number_sources)
  if(number_sources EQUAL 0)
    return()
  endif()

  # If the scratch has more than a source file, we need to find the source with
  # the main function
  set(scratch_src)
  foreach(source_file ${source_files})
    file(READ ${source_file} source_file_contents)
    string(REGEX MATCHALL "main[(| (]" main_position "${source_file_contents}")
    if(CMAKE_MATCH_0)
      set(scratch_src ${source_file})
    endif()
  endforeach()

  if(NOT scratch_src)
    return()
  endif()

  # Get parent directory name
  get_filename_component(scratch_dirname ${scratch_src} DIRECTORY)
  string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "" scratch_dirname
                 "${scratch_dirname}"
  )
  string(REPLACE "/" "_" scratch_dirname "${scratch_dirname}")

  # Get source name
  get_filename_component(scratch_name ${scratch_src} NAME_WE)

  set(target_prefix scratch_)
  if(scratch_dirname)
    # Join the names together if dirname is not the scratch folder
    set(target_prefix scratch${scratch_dirname}_)
  endif()

  # Get source absolute path and transform into relative path
  get_filename_component(scratch_src ${scratch_src} ABSOLUTE)
  get_filename_component(scratch_absolute_directory ${scratch_src} DIRECTORY)
  string(REPLACE "${PROJECT_SOURCE_DIR}" "${CMAKE_OUTPUT_DIRECTORY}"
                 scratch_directory ${scratch_absolute_directory}
  )
  build_exec(
          EXECNAME ${scratch_name}
          EXECNAME_PREFIX ${target_prefix}
          SOURCE_FILES "${source_files}"
          LIBRARIES_TO_LINK "${ns3-libs}" "${ns3-contrib-libs}" 
          EXECUTABLE_DIRECTORY_PATH ${scratch_directory}/
  )
endfunction()

# Scan *.cc files in ns-3-dev/scratch and build a target for each
file(GLOB single_source_file_scratches CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/[^.]*.cc)
foreach(scratch_src ${single_source_file_scratches})
  create_scratch(${scratch_src})
endforeach()


# if(${Torch_FOUND}) 
  # include_directories(${Torch_INCLUDE_DIRS})
  # link_libraries(${Torch_LIBRARIES})
# endif()

# Scan *.cc files in ns-3-dev/scratch subdirectories and build a target for each
# subdirectory
file(
  GLOB_RECURSE scratch_subdirectories
  CONFIGURE_DEPENDS
  LIST_DIRECTORIES true
  ${CMAKE_CURRENT_SOURCE_DIR}/**
)
# Filter out files
foreach(entry ${scratch_subdirectories})
  if(NOT (IS_DIRECTORY ${entry}))
    list(REMOVE_ITEM scratch_subdirectories ${entry})
  endif()
endforeach()

foreach(subdir ${scratch_subdirectories})
  if(EXISTS ${subdir}/CMakeLists.txt)
    # If the subdirectory contains a CMakeLists.txt file
    # we let the CMake file manage the source files
    #
    # Use this if you want to link to external libraries
    # without creating a module
    add_subdirectory(${subdir})
  else()
    # Otherwise we pick all the files in the subdirectory
    # and create a scratch for them automatically
    file(GLOB scratch_sources CONFIGURE_DEPENDS ${subdir}/[^.]*.cc)
    create_scratch("${scratch_sources}")
  endif()
endforeach()


option(DOWNLOAD_MNIST "Download the MNIST dataset from the internet" ON)
if (DOWNLOAD_MNIST)
  message(STATUS "Downloading MNIST dataset")
  execute_process(
    COMMAND python ${CMAKE_CURRENT_LIST_DIR}/tools/download_mnist.py
      -d ${CMAKE_BINARY_DIR}/data
    ERROR_VARIABLE DOWNLOAD_ERROR)
  if (DOWNLOAD_ERROR)
    message(FATAL_ERROR "Error downloading MNIST dataset: ${DOWNLOAD_ERROR}")
  endif()
endif()


# find_external_library(DEPENDENCY_NAME Torch
#                       HEADER_NAME
#                         torch/script.h
#                         torch.h
#                       LIBRARY_NAME torch
#                       SEARCH_PATHS /home/hiba/Desktop/PFE/ns-allinone-3.38/ns-3.38/contrib/libtorch)



# if(${Torch_FOUND})
#   include_directories(${Torch_INCLUDE_DIRS})
#   link_libraries(${Torch_LIBRARIES})
#   set(torch_libraries ${Torch_LIBRARIES})

  
# option(DOWNLOAD_MNIST "Download the MNIST dataset from the internet" ON)
# if (DOWNLOAD_MNIST)
#   message(STATUS "Downloading MNIST dataset")
#   execute_process(
#     COMMAND python ${CMAKE_CURRENT_LIST_DIR}/tools/download_mnist.py
#       -d ${CMAKE_BINARY_DIR}/data
#     ERROR_VARIABLE DOWNLOAD_ERROR)
#   if (DOWNLOAD_ERROR)
#     message(FATAL_ERROR "Error downloading MNIST dataset: ${DOWNLOAD_ERROR}")
#   endif()
# endif()

# endif()
