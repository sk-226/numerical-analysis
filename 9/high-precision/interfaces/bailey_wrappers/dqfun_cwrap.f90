module dqfun_cwrap
    use iso_c_binding
    use dqmodule
    implicit none
contains

subroutine dq_add(a,b,c) bind(C,name="dqadd_")
    real(dqknd), intent(in)  :: a(2), b(2)
    real(dqknd), intent(out) :: c(2)
    type(dq_real) :: da, db, dc
    da%dqr = a
    db%dqr = b
    dc = da + db
    c = dc%dqr
end subroutine

subroutine dq_sub(a,b,c) bind(C,name="dqsub_")
    real(dqknd), intent(in)  :: a(2), b(2)
    real(dqknd), intent(out) :: c(2)
    type(dq_real) :: da, db, dc
    da%dqr = a
    db%dqr = b
    dc = da - db
    c = dc%dqr
end subroutine

subroutine dq_mul(a,b,c) bind(C,name="dqmul_")
    real(dqknd), intent(in)  :: a(2), b(2)
    real(dqknd), intent(out) :: c(2)
    type(dq_real) :: da, db, dc
    da%dqr = a
    db%dqr = b
    dc = da * db
    c = dc%dqr
end subroutine

subroutine dq_div(a,b,c) bind(C,name="dqdiv_")
    real(dqknd), intent(in)  :: a(2), b(2)
    real(dqknd), intent(out) :: c(2)
    type(dq_real) :: da, db, dc
    da%dqr = a
    db%dqr = b
    dc = da / db
    c = dc%dqr
end subroutine

subroutine dq_fromdbl(d,a) bind(C,name="dqdqd_")
    real(c_double), intent(in)  :: d
    real(dqknd), intent(out) :: a(2)
    type(dq_real) :: da
    da%dqr(1) = real(d, dqknd)
    da%dqr(2) = 0.0_dqknd
    a = da%dqr
end subroutine

subroutine dq_sqrt(a,b) bind(C,name="dqsqrt_")
    real(dqknd), intent(in)  :: a(2)
    real(dqknd), intent(out) :: b(2)
    type(dq_real) :: da, db
    da%dqr = a
    db = sqrt(da)
    b = db%dqr
end subroutine

subroutine dq_tostr(a,nd,s,str_len) bind(C,name="dqtoqd_")
    real(dqknd), intent(in) :: a(2)
    integer(c_int), intent(in) :: nd
    character(c_char), intent(out) :: s(str_len)
    integer(c_int), value :: str_len
    type(dq_real) :: da
    character(len=70) :: tmp
    integer :: i, c_len
    
    da%dqr = a
    ! Convert to simple double representation to avoid dq_real formatting issues
    write(tmp, '(ES35.25E3)') real(da%dqr(1), c_double)
    c_len = min(len(trim(tmp)), str_len-1)
    do i = 1, c_len
        s(i) = tmp(i:i)
    end do
    s(c_len+1) = c_null_char
end subroutine

end module