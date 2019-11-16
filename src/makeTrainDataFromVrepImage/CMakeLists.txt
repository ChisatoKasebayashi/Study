project(makeTrainDataFromVrepImage)

find_package(OpenCV)
find_package(Boost COMPONENTS system filesystem REQUIRED)
include_directories(
    ${Boost_INCLUDE_DIR}
    )
link_directories(
	${Boost_LIBRARY_DIR}
    )
add_executable(makeTrainData
    makeTrainDataFromVrepImage.cpp
)
target_link_libraries(makeTrainData
	${OpenCV_LIBS}
    ${Boost_LIBRARIES}
)
