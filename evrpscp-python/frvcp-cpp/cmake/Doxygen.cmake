if(${PROJECT_NAME}_ENABLE_DOXYGEN)
    set(DOXYGEN_CALLER_GRAPH YES)
    set(DOXYGEN_CALL_GRAPH YES)
    set(DOXYGEN_EXTRACT_ALL YES)
    set(DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/docs)
    set(DOXYGEN_EXCLUDE_PATTERNS "*/lib/*")

    find_package(Doxygen REQUIRED dot)
    doxygen_add_docs(doxygen-docs ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src)

    verbose_message("Doxygen has been setup and documentation is now available.")
endif()
