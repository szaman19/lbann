# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  factory.cpp
  kfac.cpp
  ltfb.cpp
  sgd_training_algorithm.cpp
  training_algorithm.cpp
  )

# Add the support class implementations
add_subdirectory(kfac)
add_subdirectory(ltfb)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(GPU_SOURCES "${GPU_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
