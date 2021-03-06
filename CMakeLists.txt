CMAKE_MINIMUM_REQUIRED(VERSION 3.10.2)
IF(COMMAND CMAKE_POLICY)
  CMAKE_POLICY(SET CMP0003 NEW)
ENDIF(COMMAND CMAKE_POLICY)

SET(PCA PrincipalComponentsAnalysis)
PROJECT(${PCA})

IF(NOT ITK_SOURCE_DIR)
  FIND_PACKAGE(ITK REQUIRED)
  LIST(APPEND CMAKE_MODULE_PATH ${ITK_CMAKE_DIR})
  INCLUDE(ITKModuleExternal)
ELSE()
  itk_module_impl()
ENDIF()

CMAKE_DEPENDENT_OPTION(Module_${PCA}_BUILD_EXAMPLES "Build the examples" OFF "BUILD_EXAMPLES" OFF)
if(Module_${PCA}_BUILD_EXAMPLES)
  ADD_SUBDIRECTORY( examples )
ENDIF()


CMAKE_DEPENDENT_OPTION(Module_${PCA}_BUILD_DOCUMENTATION "Generate documentation from LaTeX files, source code examples and result screenshots" OFF
	"BUILD_DOCUMENTATION" OFF)
IF(Module_${PCA}_BUILD_DOCUMENTATION)
  ADD_SUBDIRECTORY( doc )
ENDIF()
