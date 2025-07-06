cmake_minimum_required(VERSION 3.21)
project(bailey_hp LANGUAGES C CXX Fortran)

# ---------- コンパイラ最適化設定 -----------------------------------------------
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto -DNDEBUG")
set(CMAKE_C_FLAGS_RELEASE "-O3 -march=native -flto -DNDEBUG")
set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -march=native -flto")

# ---------- 依存ライブラリの設定 -----------------------------------------------
find_package(Eigen3 3.4 REQUIRED)

# ---------- BLAS/LAPACK統合 (double精度高速化) ------------------------------
find_package(BLAS)
find_package(LAPACK)

if(BLAS_FOUND AND LAPACK_FOUND)
    message(STATUS "BLAS library found: ${BLAS_LIBRARIES}")
    message(STATUS "LAPACK library found: ${LAPACK_LIBRARIES}")
    set(BLAS_ENABLED TRUE)
else()
    message(WARNING "BLAS/LAPACK not found - using Eigen's default implementations")
    set(BLAS_ENABLED FALSE)
endif()

# ---------- fast_matrix_market ライブラリを追加 ------------------------------
add_subdirectory(lib/fast_matrix_market EXCLUDE_FROM_ALL)

# ---------- 各Bailey精度ライブラリを IMPORTED で登録 -------------------------
foreach(name dd dq qx)
    string(TOUPPER ${name} NAME_UP)
    set(${NAME_UP}FUN_DIR $ENV{${NAME_UP}FUN_DIR})
    add_library(${name}fun STATIC IMPORTED)
    set_target_properties(${name}fun PROPERTIES
        IMPORTED_LOCATION "${${NAME_UP}FUN_DIR}/lib${name}fun.a")
    add_library(${name}wrap STATIC IMPORTED)
    set_target_properties(${name}wrap PROPERTIES
        IMPORTED_LOCATION "${${NAME_UP}FUN_DIR}/lib${name}wrap.a")
endforeach()

# ---------- 共通のインクルードディレクトリとリンクライブラリ ------------------
set(COMMON_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    $ENV{QXFUN_DIR}
    $ENV{DQFUN_DIR}
    $ENV{DDFUN_DIR}
    ${CMAKE_SOURCE_DIR}/lib/fast_matrix_market/include
)

set(COMMON_LIBRARIES qxwrap qxfun dqwrap dqfun ddwrap ddfun gfortran)
if(BLAS_ENABLED)
    set(MATRIX_MARKET_LIBRARIES ${COMMON_LIBRARIES} fast_matrix_market ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
else()
    set(MATRIX_MARKET_LIBRARIES ${COMMON_LIBRARIES} fast_matrix_market)
endif()

# ---------- 実験用プログラム --------------------------------------------------

# メインテンプレート化CG Solver
add_executable(cg_solver src/cg_solver.cpp)
target_include_directories(cg_solver PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(cg_solver PRIVATE ${MATRIX_MARKET_LIBRARIES})
target_compile_features(cg_solver PRIVATE cxx_std_20)
if(BLAS_ENABLED)
    target_compile_definitions(cg_solver PRIVATE EIGEN_USE_BLAS EIGEN_USE_LAPACKE)
endif()

# メインデモプログラム
add_executable(sample_qx src/experiments/main.cpp)
target_include_directories(sample_qx PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(sample_qx PRIVATE ${MATRIX_MARKET_LIBRARIES})
target_compile_features(sample_qx PRIVATE cxx_std_17)

# nos7 CG実験プログラム
add_executable(nos7_cg src/experiments/nos7_cg.cpp)
target_include_directories(nos7_cg PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(nos7_cg PRIVATE ${MATRIX_MARKET_LIBRARIES})
target_compile_features(nos7_cg PRIVATE cxx_std_17)

# nos5 CG実験プログラム
add_executable(nos5_cg src/experiments/nos5_cg.cpp)
target_include_directories(nos5_cg PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(nos5_cg PRIVATE ${MATRIX_MARKET_LIBRARIES})
target_compile_features(nos5_cg PRIVATE cxx_std_17)

# ---------- テスト・ベンチマークプログラム -----------------------------------

# 基本的なQuadDouble演算テスト
add_executable(test_basic src/benchmarks/test_basic.cpp)
target_include_directories(test_basic PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(test_basic PRIVATE ${COMMON_LIBRARIES})
target_compile_features(test_basic PRIVATE cxx_std_17)

# Simple matrix market test
add_executable(simple_test src/simple_test.cpp)
target_include_directories(simple_test PRIVATE ${COMMON_INCLUDE_DIRS})
target_link_libraries(simple_test PRIVATE ${COMMON_LIBRARIES})
target_compile_features(simple_test PRIVATE cxx_std_17)

# ---------- Fortranラッパーの構築 ---------------------------------------------
# 注意: Fortranラッパーはinterfaces/bailey_wrappers/に移動済み
# 実際のビルドでは環境変数で指定されたBaileyライブラリを使用