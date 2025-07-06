module ddfun_cwrap
    use iso_c_binding
    use ddmodule
    implicit none
contains

subroutine dd_add(a,b,c) bind(C,name="ddadd_")
    real(c_double), intent(in)  :: a(2), b(2)
    real(c_double), intent(out) :: c(2)
    type(dd_real) :: da, db, dc
    da%ddr = a
    db%ddr = b
    dc = da + db
    c = dc%ddr
end subroutine

subroutine dd_sub(a,b,c) bind(C,name="ddsub_")
    real(c_double), intent(in)  :: a(2), b(2)
    real(c_double), intent(out) :: c(2)
    type(dd_real) :: da, db, dc
    da%ddr = a
    db%ddr = b
    dc = da - db
    c = dc%ddr
end subroutine

subroutine dd_mul(a,b,c) bind(C,name="ddmul_")
    real(c_double), intent(in)  :: a(2), b(2)
    real(c_double), intent(out) :: c(2)
    type(dd_real) :: da, db, dc
    da%ddr = a
    db%ddr = b
    dc = da * db
    c = dc%ddr
end subroutine

subroutine dd_div(a,b,c) bind(C,name="dddiv_")
    real(c_double), intent(in)  :: a(2), b(2)
    real(c_double), intent(out) :: c(2)
    type(dd_real) :: da, db, dc
    da%ddr = a
    db%ddr = b
    dc = da / db
    c = dc%ddr
end subroutine

subroutine dd_fromdbl(d,a) bind(C,name="dddqd_")
    real(c_double), intent(in)  :: d
    real(c_double), intent(out) :: a(2)
    type(dd_real) :: da
    da%ddr(1) = d
    da%ddr(2) = 0.0d0
    a = da%ddr
end subroutine

subroutine dd_sqrt(a,b) bind(C,name="ddsqrt_")
    real(c_double), intent(in)  :: a(2)
    real(c_double), intent(out) :: b(2)
    type(dd_real) :: da, db
    da%ddr = a
    db = sqrt(da)
    b = db%ddr
end subroutine

subroutine dd_tostr(a,nd,s,str_len) bind(C,name="ddtoqd_")
    real(c_double), intent(in) :: a(2)
    integer(c_int), intent(in) :: nd
    character(c_char), intent(out) :: s(str_len)
    integer(c_int), value :: str_len
    type(dd_real) :: da
    character(len=70) :: tmp
    integer :: i, c_len
    
    da%ddr = a
    ! Convert to simple double representation to avoid dd_real formatting issues
    write(tmp, '(ES25.15E3)') da%ddr(1)
    c_len = min(len(trim(tmp)), str_len-1)
    do i = 1, c_len
        s(i) = tmp(i:i)
    end do
    s(c_len+1) = c_null_char
end subroutine

end module