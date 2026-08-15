# Apply a Git patch exactly once to an ExternalProject source tree.

foreach(required_variable GIT_EXECUTABLE SOURCE_DIR PATCH_FILE)
  if(NOT DEFINED ${required_variable})
    message(FATAL_ERROR "${required_variable} is required")
  endif()
endforeach()

execute_process(
  COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" apply --reverse --check
          --unidiff-zero --whitespace=nowarn "${PATCH_FILE}"
  RESULT_VARIABLE reverse_check_result
  OUTPUT_QUIET
  ERROR_QUIET
)
if(reverse_check_result EQUAL 0)
  message(STATUS "Git patch is already applied: ${PATCH_FILE}")
  return()
endif()

execute_process(
  COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" apply --check
          --unidiff-zero --whitespace=nowarn "${PATCH_FILE}"
  RESULT_VARIABLE apply_check_result
  OUTPUT_VARIABLE apply_check_output
  ERROR_VARIABLE apply_check_error
)
if(NOT apply_check_result EQUAL 0)
  message(FATAL_ERROR
    "Git patch cannot be applied to ${SOURCE_DIR}: ${PATCH_FILE}\n"
    "${apply_check_output}${apply_check_error}"
  )
endif()

execute_process(
  COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" apply
          --unidiff-zero --whitespace=nowarn "${PATCH_FILE}"
  RESULT_VARIABLE apply_result
  OUTPUT_VARIABLE apply_output
  ERROR_VARIABLE apply_error
)
if(NOT apply_result EQUAL 0)
  message(FATAL_ERROR
    "Git patch application failed in ${SOURCE_DIR}: ${PATCH_FILE}\n"
    "${apply_output}${apply_error}"
  )
endif()
