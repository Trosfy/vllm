if(NOT DEFINED SOURCE_DIR OR NOT DEFINED PATCH_FILE)
  message(FATAL_ERROR "SOURCE_DIR and PATCH_FILE are required")
endif()

execute_process(
  COMMAND git apply --unidiff-zero --check "${PATCH_FILE}"
  WORKING_DIRECTORY "${SOURCE_DIR}"
  RESULT_VARIABLE PATCH_APPLIES
  OUTPUT_QUIET
  ERROR_QUIET)

if(PATCH_APPLIES EQUAL 0)
  execute_process(
    COMMAND git apply --unidiff-zero --whitespace=nowarn "${PATCH_FILE}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE PATCH_RESULT
    OUTPUT_VARIABLE PATCH_OUTPUT
    ERROR_VARIABLE PATCH_ERROR)
  if(NOT PATCH_RESULT EQUAL 0)
    message(FATAL_ERROR
      "Failed to apply ${PATCH_FILE}:\n${PATCH_OUTPUT}${PATCH_ERROR}")
  endif()
  message(STATUS "Applied ${PATCH_FILE}")
else()
  execute_process(
    COMMAND git apply --unidiff-zero --reverse --check "${PATCH_FILE}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE PATCH_ALREADY_APPLIED
    OUTPUT_QUIET
    ERROR_QUIET)
  if(NOT PATCH_ALREADY_APPLIED EQUAL 0)
    message(FATAL_ERROR
      "${PATCH_FILE} neither applies nor is already applied in ${SOURCE_DIR}")
  endif()
  message(STATUS "Patch already applied: ${PATCH_FILE}")
endif()
