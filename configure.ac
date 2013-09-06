AC_PREREQ(2.5)
AC_INIT([tuneit], [0.3], [Mario Lang <mlang@delysid.org>])
AM_INIT_AUTOMAKE
AC_CONFIG_SRCDIR([src/tuneit.c])
AM_CONFIG_HEADER(config.h)
AC_PROG_CC
AC_PROG_INSTALL
AC_CHECK_LIB([asound], [snd_pcm_open], , [AC_MSG_ERROR([Couldn't find ALSA soundcard drivers. You must install these first. See www.alsa-project.org or use ALSA modules in linux kernel version 2.6])])
AC_CHECK_LIB([fftw3f], [fftwf_plan_dft_r2c_1d], , [AC_MSG_ERROR([Couldn't find FFTW3 library. You must install it first.])])
AC_CHECK_LIB([jack], [jack_client_new], , [AC_MSG_ERROR([Couldn't find JACK library. You must install it first.])])
AC_CHECK_HEADER([alsa/asoundlib.h], , [AC_MSG_ERROR([Couldn't find alsa header files. You must install this first. Debian users apt-get install alsa-headers])])
AC_OUTPUT(Makefile src/Makefile)

