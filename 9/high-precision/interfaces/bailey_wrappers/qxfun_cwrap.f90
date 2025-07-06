module qxfun_cwrap
    use iso_c_binding
    use qxmodule
    implicit none
contains

subroutine qx_add(a,b,c) bind(C,name="qxadd_")
    real(qxknd), intent(in)  :: a, b
    real(qxknd), intent(out) :: c
    c = a + b
end subroutine

subroutine qx_sub(a,b,c) bind(C,name="qxsub_")
    real(qxknd), intent(in)  :: a, b
    real(qxknd), intent(out) :: c
    c = a - b
end subroutine

subroutine qx_mul(a,b,c) bind(C,name="qxmul_")
    real(qxknd), intent(in)  :: a, b
    real(qxknd), intent(out) :: c
    c = a * b
end subroutine

subroutine qx_div(a,b,c) bind(C,name="qxdiv_")
    real(qxknd), intent(in)  :: a, b
    real(qxknd), intent(out) :: c
    c = a / b
end subroutine

subroutine qx_fromdbl(d,a) bind(C,name="qxdqd_")
    real(c_double), intent(in)  :: d
    real(qxknd), intent(out) :: a
    a = real(d, qxknd)
end subroutine

subroutine qx_sqrt(a,b) bind(C,name="qxsqrt_")
    real(qxknd), intent(in)  :: a
    real(qxknd), intent(out) :: b
    b = sqrt(a)
end subroutine

subroutine qx_tostr(a,nd,s,str_len) bind(C,name="qxtoqd_")
    real(qxknd), intent(in) :: a
    integer(c_int), intent(in) :: nd
    character(c_char), intent(out) :: s(str_len)
    integer(c_int), value :: str_len
    character(len=70) :: tmp
    character(len=10) :: fmt_str
    integer :: i, c_len
    
    write(fmt_str, '(I0)') nd
    write(tmp, '(ES0.' // trim(fmt_str) // ')') a
    c_len = min(len(trim(tmp)), str_len-1)
    do i = 1, c_len
        s(i) = tmp(i:i)
    end do
    s(c_len+1) = c_null_char
end subroutine

end module