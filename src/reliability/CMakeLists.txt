build_lib(
  LIBNAME reliability
  SOURCE_FILES
    model/power-model.cc
    model/util-power-model.cc
    model/app-power-model.cc
    model/temperature-model.cc
    model/temperature-simple-model.cc      
    model/performance-model.cc
    model/performance-simple-model.cc
    model/reliability-model.cc
    model/reliability-tddb-model.cc
    helper/reliability-helper.cc
  HEADER_FILES
    model/power-model.h
    model/util-power-model.h
    model/app-power-model.h
    model/temperature-model.h
    model/temperature-simple-model.h
    model/performance-model.h
    model/performance-simple-model.h
    model/reliability-model.h
    model/reliability-tddb-model.h
    helper/reliability-helper.h
  LIBRARIES_TO_LINK ${libcore}
)

