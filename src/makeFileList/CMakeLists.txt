project(ReadFileExample)

find_package(OpenCV)
find_package(Boost COMPONENTS system filesystem REQUIRED)
include_directories(
    ${Boost_INCLUDE_DIR}
    )
link_directories(
	${Boost_LIBRARY_DIR}
    )
add_executable(makeFileList
    makeFileList.cpp
)
target_link_libraries(makeFileList
	${OpenCV_LIBS}
    ${Boost_LIBRARIES}
)
