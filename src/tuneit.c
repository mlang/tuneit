/* tuneit.c -- Detect fundamental frequency of a sound
 * Copyright (C) 2004, 2005  Mario Lang <mlang@delysid.org>
 *
 * This is free software, placed under the terms of the
 * GNU General Public License, as published by the Free Software Foundation.
 * Please see the file COPYING for details.
 */
#include "config.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static unsigned int rate = 48000;

/* Constants */

/* pow(2.0,1.0/12.0) == 100 cents == 1 half-tone */
#define D_NOTE          1.059463094359
/* log(pow(2.0,1.0/12.0)) */
#define LOG_D_NOTE      0.057762265047
/* pow(2.0,1.0/24.0) == 50 cents */
#define D_NOTE_SQRT     1.029302236643
/* log(2) */ 
#define LOG_2           0.693147180559

static double freqs[12];
static double lfreqs[12];

static const char *englishNotes[12] = {"A","A#","B","C","C#","D","D#","E","F","F#", "G", "G#"};
static const char *germanNotes[12] = {"A","A#","H","C","C#","D","D#","E","F","F#", "G", "G#"};
static const char *frenchNotes[12] = {"La","La#","Si","Do","Do#","Ré","Ré#",
				 "Mi","Fa","Fa#","Sol","Sol#"};
static const char **notes = englishNotes;

static void
displayFrequency (double freq)
{
  double ldf, mldf;
  double lfreq, nfreq;
  int i, note = 0;

  if (freq < 1E-15) freq = 1E-15;
  lfreq = log(freq);
  while (lfreq < lfreqs[0]-LOG_D_NOTE/2.) lfreq += LOG_2;
  while (lfreq >= lfreqs[0]+LOG_2-LOG_D_NOTE/2.) lfreq -= LOG_2;
  mldf = LOG_D_NOTE;
  for (i=0; i<12; i++) {
    ldf = fabs(lfreq-lfreqs[i]);
    if (ldf < mldf) {
      mldf = ldf;
      note = i;
    }
  }
  nfreq = freqs[note];
  while (nfreq/freq > D_NOTE_SQRT) nfreq /= 2.0;
  while (freq/nfreq > D_NOTE_SQRT) nfreq *= 2.0;
  printf("Note %-2s (%8.3fHz): %+3.f cents (%8.3fHz)      \r",
	 notes[note], nfreq, 1200*(log(freq/nfreq)/LOG_2), freq);
  fflush(stdout);
}

typedef struct {
  void (*init) (int);
  void (*measures16) (int, signed short int *);
  void (*measurefloat) (int, float *);
  void (*free) (void);
} MeasureAlgorithm;
static const MeasureAlgorithm *algorithm = NULL;

int blockSize;
static signed short int *schmittBuffer = NULL;
static signed short int *schmittPointer = NULL;

static void
schmittInit (int size)
{
  blockSize = rate/size;
  schmittBuffer = (signed short int *)malloc(blockSize*sizeof(signed short int));
  schmittPointer = schmittBuffer;
}

static inline int
Abs(int x)
{
  return ((x>0) ? x : -x);
}

static void
schmittS16LE (int nframes, signed short int *indata)
{
  int i, j;
  double trigfact = 0.6;

  for (i=0; i<nframes; i++) {
    *schmittPointer++ = indata[i];
    if (schmittPointer-schmittBuffer >= blockSize) {
      int endpoint, startpoint, t1, t2, A1, A2, tc, schmittTriggered;

      schmittPointer = schmittBuffer;

      for (j=0,A1=0,A2=0; j<blockSize; j++) {
	if (schmittBuffer[j]>0 && A1<schmittBuffer[j])  A1 = schmittBuffer[j];
	if (schmittBuffer[j]<0 && A2<-schmittBuffer[j]) A2 = -schmittBuffer[j];
      }
      t1 =   (int)( A1 * trigfact + 0.5);
      t2 = - (int)( A2 * trigfact + 0.5);
      startpoint=0;
      for (j=1; schmittBuffer[j]<=t1 && j<blockSize; j++);
      for (; !(schmittBuffer[j]  >=t2 &&
	       schmittBuffer[j+1]< t2) && j<blockSize; j++);
      startpoint=j;
      schmittTriggered=0;
      endpoint=startpoint+1;
      for(j=startpoint,tc=0; j<blockSize; j++) {
	if (!schmittTriggered) {
	  schmittTriggered = (schmittBuffer[j] >= t1);
	} else if (schmittBuffer[j]>=t2 && schmittBuffer[j+1]<t2) {
	  endpoint=j;
	  tc++;
	  schmittTriggered = 0;
	}
      }
      if (endpoint > startpoint) {
	displayFrequency((double)rate*(tc/(double)(endpoint-startpoint)));
      }
    }
  }
}

static void
schmittFloat (int nframes, float *indata)
{
  signed short int buf[nframes];
  int i;
  for (i=0; i<nframes; i++) {
    buf[i] = indata[i]*32768.;
  }
  schmittS16LE(nframes, buf);
}

static void
schmittFree ()
{
  free(schmittBuffer);
}

static const MeasureAlgorithm schmittTriggerAlgorithm = {
  schmittInit, schmittS16LE, schmittFloat, schmittFree
};

#include <complex.h>
#include <fftw3.h>
#include <string.h>

#define M_PI 3.14159265358979323846
#define MAX_FFT_LENGTH 48000
static float *fftSampleBuffer;
static float *fftSample;
static float *fftLastPhase;
static int fftSize;
static int fftFrameCount = 0;
static float *fftIn;
static fftwf_complex *fftOut;
static fftwf_plan fftPlan;

typedef struct {
  double freq;
  double db;
} Peak;
#define MAX_PEAKS 8

static void
fftInit (int size)
{
  int i;

  fftSize = rate/size;
  fftIn = fftwf_malloc(sizeof(float) * 2 * (fftSize/2+1));
  fftOut = (fftwf_complex *)fftIn;
  fftPlan = fftwf_plan_dft_r2c_1d(fftSize, fftIn, fftOut, FFTW_MEASURE);

  fftSampleBuffer = (float *)malloc(fftSize * sizeof(float));
  fftSample = NULL;
  fftLastPhase = (float *)malloc((fftSize/2+1) * sizeof(float));
  memset(fftSampleBuffer, 0, fftSize*sizeof(float));
  memset(fftLastPhase, 0, (fftSize/2+1)*sizeof(float));
}

static void
fftMeasure (int nframes, int overlap, float *indata)
{
  int i, stepSize = fftSize/overlap;
  double freqPerBin = rate/(double)fftSize,
    phaseDifference = 2.*M_PI*(double)stepSize/(double)fftSize;

  if (!fftSample) fftSample = fftSampleBuffer + (fftSize-stepSize);

  for (i=0; i<nframes; i++) {
    *fftSample++ = indata[i];
    if (fftSample-fftSampleBuffer >= fftSize) {
      int k;
      Peak peaks[MAX_PEAKS];

      for (k=0; k<MAX_PEAKS; k++) {
	peaks[k].db = -200.;
	peaks[k].freq = 0.;
      }

      fftSample = fftSampleBuffer + (fftSize-stepSize);

      for (k=0; k<fftSize; k++) {
        double window = -.5*cos(2.*M_PI*(double)k/(double)fftSize)+.5;
        fftIn[k] = fftSampleBuffer[k] * window;
      }
      fftwf_execute(fftPlan);

      for (k=0; k<=fftSize/2; k++) {
	long qpd;
	float
	  real = creal(fftOut[k]),
	  imag = cimag(fftOut[k]),
	  magnitude = 20.*log10(2.*sqrt(real*real + imag*imag)/fftSize),
	  phase = atan2(imag, real),
	  tmp, freq;

        /* compute phase difference */
        tmp = phase - fftLastPhase[k];
        fftLastPhase[k] = phase;

        /* subtract expected phase difference */
        tmp -= (double)k*phaseDifference;

        /* map delta phase into +/- Pi interval */
        qpd = tmp / M_PI;
        if (qpd >= 0) qpd += qpd&1;
        else qpd -= qpd&1;
        tmp -= M_PI*(double)qpd;

        /* get deviation from bin frequency from the +/- Pi interval */
        tmp = overlap*tmp/(2.*M_PI);

        /* compute the k-th partials' true frequency */
        freq = (double)k*freqPerBin + tmp*freqPerBin;

	if (freq > 0.0 && magnitude > peaks[0].db) {
	  memmove(peaks+1, peaks, sizeof(Peak)*(MAX_PEAKS-1));
	  peaks[0].freq = freq;
	  peaks[0].db = magnitude;
	}
      }
      fftFrameCount++;
      if (fftFrameCount > 0 && fftFrameCount % overlap == 0) {
	int l, maxharm = 0;
	k = 0;
	for (l=1; l<MAX_PEAKS && peaks[l].freq > 0.0; l++) {
	  int harmonic;

	  for (harmonic=5; harmonic>1; harmonic--) {
	    if (peaks[0].freq / peaks[l].freq < harmonic+.02 &&
		peaks[0].freq / peaks[l].freq > harmonic-.02) {
	      if (harmonic > maxharm &&
		  peaks[0].db < peaks[l].db/2) {
		maxharm = harmonic;
		k = l;
	      }
	    }
	  }
	}
	displayFrequency(peaks[k].freq);
      }
      memmove(fftSampleBuffer, fftSampleBuffer+stepSize, (fftSize-stepSize)*sizeof(float));
    }
  }
}

static void
fftFloat (int nframes, float *indata)
{
  fftMeasure(nframes, 4, indata);
}

static void
fftS16LE (int nframes, signed short int *indata)
{
  float buf[nframes];
  int i;
  for (i=0; i<nframes; i++) {
    buf[i] = indata[i]/32768.;
  }
  fftMeasure(nframes, 4, buf);
}

static void
fftFree ()
{
  fftwf_destroy_plan(fftPlan);
  fftwf_free(fftIn);
  free(fftSampleBuffer);
}

static const MeasureAlgorithm fftAlgorithm = {
  fftInit, fftS16LE, fftFloat, fftFree
};

typedef struct {
  void (*init) (void);
  void (*listPorts) (void);
  void (*open) (char *);
  void (*run) (void);
  void (*close) (void);
  void (*free) (void);
} AudioInterface;
static const AudioInterface *audio = NULL;

#include <alsa/asoundlib.h>

static snd_pcm_t *alsaHandle;

static void
alsaInit ()
{
}

/* Helper macro for common ALSA error checking code */
#define DO_OR_DIE(a,b) if ((result = (a)) < 0) { \
                         fprintf(stderr, b ": %s\n", snd_strerror(result)); \
                         exit(EXIT_FAILURE); \
                       }
static void
alsaListPorts ()
{
  int cardIndex = -1;
  snd_ctl_card_info_t *info;
  snd_pcm_info_t *pcminfo;

  snd_ctl_card_info_malloc(&info);
  snd_pcm_info_malloc(&pcminfo);

  while (snd_card_next(&cardIndex) == 0 && cardIndex >= 0) {
    snd_ctl_t *ctlHandle;
    char str[128];
    int result;
    
    sprintf(str, "hw:CARD=%i", cardIndex);
    if ((result = snd_ctl_open(&ctlHandle, str, 0)) >= 0) {
      if ((result = snd_ctl_card_info(ctlHandle, info)) >= 0) {
	int deviceIndex = -1;

	while (snd_ctl_pcm_next_device(ctlHandle, &deviceIndex) == 0 &&
	       deviceIndex >= 0) {
	  snd_pcm_info_set_device(pcminfo, deviceIndex);
	  snd_pcm_info_set_subdevice(pcminfo, 0);
	  snd_pcm_info_set_stream(pcminfo, SND_PCM_STREAM_CAPTURE);
	  if ((result = snd_ctl_pcm_info(ctlHandle, pcminfo)) >= 0) {
	    printf("hw:%d,%d\t%s\n",
		   snd_pcm_info_get_card(pcminfo),
		   snd_pcm_info_get_device(pcminfo),
		   snd_pcm_info_get_name(pcminfo));
	  }
	}
      } else {
	fprintf(stderr, "Cannot aquire HW info: %s\n", snd_strerror(result));
      }

      snd_ctl_close(ctlHandle);
    } else {
      fprintf(stderr, "Cannot open mixer for %s: %s\n",
	      str, snd_strerror(result));
    }
  }
  snd_ctl_card_info_free(info);
  snd_pcm_info_free(pcminfo);
}

static void
alsaOpen (char *captureDevice)
{
  snd_pcm_hw_params_t *hw_params;
  char *deviceName;
  int result;

  if (captureDevice && captureDevice[0]) {
    deviceName = captureDevice;
  } else {
    deviceName = "hw:0,0";
  }
  if ((result = snd_pcm_open(&alsaHandle, deviceName,
			     SND_PCM_STREAM_CAPTURE, 0)) < 0) {
    fprintf(stderr, "Cannot open audio device %s: %s\n",
	    deviceName, snd_strerror(result));
    exit(EXIT_FAILURE);
  }
  DO_OR_DIE(snd_pcm_hw_params_malloc(&hw_params),
	    "Cannot allocate hardware parameter structure");
  DO_OR_DIE(snd_pcm_hw_params_any(alsaHandle, hw_params),
	    "Cannot initialize hardware parameter structure");
  DO_OR_DIE(snd_pcm_hw_params_set_access(alsaHandle, hw_params,
					 SND_PCM_ACCESS_RW_INTERLEAVED),
	    "Cannot set access type");
  DO_OR_DIE(snd_pcm_hw_params_set_format(alsaHandle, hw_params,
					 SND_PCM_FORMAT_S16_LE),
	    "Cannot set 16 bit signed integer (little-endian) sample format");
  DO_OR_DIE(snd_pcm_hw_params_set_rate_near(alsaHandle, hw_params,
					    &rate, 0),
	    "Cannot set sample rate");
  DO_OR_DIE(snd_pcm_hw_params_set_channels(alsaHandle, hw_params, 1),
	    "Cannot set channel count (mono)");
  DO_OR_DIE(snd_pcm_hw_params(alsaHandle, hw_params),
	    "Cannot set hardware parameters");
  snd_pcm_hw_params_free(hw_params);
}

static void
alsaRun ()
{
  int result;
  int nFrames = 0;
  signed short int buf[4096];

  DO_OR_DIE(snd_pcm_prepare(alsaHandle),
	    "Cannot prepare ALSA audio interface");

  while ((nFrames = snd_pcm_readi(alsaHandle, buf, 512)) > 0) {
    algorithm->measures16(nFrames, buf);
  }

  printf("\nALSA error: %s\n", snd_strerror(nFrames));
}

static void
alsaClose ()
{
  snd_pcm_close(alsaHandle);
}

static void
alsaFree ()
{
}

static const AudioInterface alsaInterface = {
  alsaInit, alsaListPorts, alsaOpen, alsaRun, alsaClose, alsaFree
};

#include <jack/jack.h>
#include <jack/ringbuffer.h>

static jack_client_t *jackClient;
static jack_port_t *jackPort;
static char *jackSourceName = NULL;
static jack_ringbuffer_t *ringBuffer;
static int jackCanProcess = 0;
static int jackOverruns = 0;

int
jackProcess (jack_nframes_t nframes, void *arg)
{
  int i;
  jack_default_audio_sample_t *in;

  if (!jackCanProcess) return 0;

  in = jack_port_get_buffer(jackPort, nframes);

  for (i=0; i<nframes; i++)
    if (jack_ringbuffer_write(ringBuffer, (void *)(in+i),
			      sizeof(jack_default_audio_sample_t))
	< sizeof(jack_default_audio_sample_t)) {
      printf("overrun!\n");
      jackOverruns++;
    }

  return 0;
}

void
jackShutdown (void *arg)
{
  fprintf(stderr, "\nJACK shutdown\n");
  exit(EXIT_FAILURE);
}

static void
jackInit ()
{
  ringBuffer = jack_ringbuffer_create(sizeof(jack_default_audio_sample_t) * 16384);
  memset(ringBuffer->buf, 0, ringBuffer->size);
  if ((jackClient = jack_client_new(PACKAGE_NAME)) == 0) {
    fprintf(stderr, "JACK server not running?\n");
    exit(EXIT_FAILURE);
  }
  jack_set_process_callback(jackClient, jackProcess, NULL);
  jack_on_shutdown(jackClient, jackShutdown, NULL);
}

static void
jackListPorts ()
{
  int i;
  const char **ports = jack_get_ports(jackClient, NULL, NULL, 0);

  for (i=0; ports[i]; i++) {
    jack_port_t *port = jack_port_by_name(jackClient, ports[i]);

    if (port && jack_port_flags(port) & JackPortIsOutput)
      printf("%s\n", ports[i]);
  }
}

static void
jackOpen (char *source_name)
{
  rate = jack_get_sample_rate(jackClient);
  if (jack_activate(jackClient)) {
    fprintf(stderr, "Cannot activate client\n");
    exit(EXIT_FAILURE);
  }
  if ((jackPort = jack_port_register(jackClient, "input",
				     JACK_DEFAULT_AUDIO_TYPE, JackPortIsInput,
				     0))
      == 0) {
    fprintf(stderr, "Cannot register JACK input port \"%s\"!\n",
	     "input");
    jack_client_close(jackClient);
    exit (EXIT_FAILURE);
  }
  if (source_name && source_name[0]) {
    jackSourceName = source_name;
  } else {
    const char **ports;
    if ((ports = jack_get_ports(jackClient, NULL, NULL, JackPortIsOutput))
	== NULL) {
      fprintf(stderr, "Cannot find any capture ports\n");
      exit(EXIT_FAILURE);
    }
    jackSourceName = strdup(ports[0]);
  }
}

static void
jackRun ()
{
  if (jack_connect(jackClient, jackSourceName, jack_port_name(jackPort))) {
    fprintf (stderr, "Cannot connect input port %s to %s\n",
	     jack_port_name(jackPort), jackSourceName);
    jack_client_close(jackClient);
    exit(EXIT_FAILURE);
  }
  jackCanProcess = 1;
  while (1) {
    if (jack_ringbuffer_read_space(ringBuffer)
	>= sizeof(jack_default_audio_sample_t)) {
      jack_default_audio_sample_t jsample;
      float sample;
      jack_ringbuffer_read(ringBuffer, (void *)&jsample, sizeof(jack_default_audio_sample_t));
      sample = jsample;
      algorithm->measurefloat(1, &sample);
    } else {
      usleep(1);
    }
  }
}

static void
jackClose ()
{
}

static void
jackFree ()
{
  jack_client_close(jackClient);
  jack_ringbuffer_free(ringBuffer);
}

static const AudioInterface jackInterface = {
  jackInit, jackListPorts, jackOpen, jackRun, jackClose, jackFree
};

int main(int argc, char *argv[])
{
  char *captureDevice = NULL;
  double aFreq = 440.0;
  int listAndExit = 0, latency = 10;
  int c;

  audio = &alsaInterface;
  algorithm = &schmittTriggerAlgorithm;
  while ((c = getopt(argc, argv, "fijl:r:t:")) != -1) {
    switch (c) {
      case 'f':
	algorithm = &fftAlgorithm;
	break;
      case 'i':
	listAndExit = 1;
	break;
      case 'j':
	audio = &jackInterface;
	break;
      case 'l':
	latency = atoi(optarg);
	break;
      case 'r':
	rate = atoi(optarg);
	break;
      case 't':
	aFreq = atof(optarg);
	break;
      default:
	fprintf(stderr, "%s [OPTIONS...] [captureDevice]\n", argv[0]);
	fprintf(stderr, "Valid options:\n");
	fprintf(stderr, "\t-f\t\tUse the more CPU intensive FFT based algorithm\n");
	fprintf(stderr, "\t-i\t\tList available input ports and exit\n");
	fprintf(stderr, "\t-j\t\tUse JACK as the audio transport system\n");
	fprintf(stderr, "\t-l LATENCY\tMeasurement window size in 1/N seconds (default is 10)\n");
	fprintf(stderr, "\t-r RATE\t\tSet sample rate (default is 48000)\n");
	fprintf(stderr, "\t-t HERTZ\tTune the A note of the scale (default is 440.0)\n");
	exit(EXIT_FAILURE);
    }
  }
  if (optind < argc) {
    captureDevice = argv[optind++];
  }
  if (optind < argc) {
    fprintf(stderr, "You can specify only one capture device\n");
    exit(EXIT_FAILURE);
  }
  /* Initialize tuning */
  {
    int i;
    freqs[0]=aFreq;
    lfreqs[0]=log(freqs[0]);
    for (i=1; i<12; i++) {
      freqs[i] = freqs[i-1] * D_NOTE;
      lfreqs[i] = lfreqs[i-1] + LOG_D_NOTE;
    }
  }
  audio->init();
  if (listAndExit) {
    audio->listPorts();
  } else {
    audio->open(captureDevice);
    algorithm->init(latency);
    audio->run();
    audio->close();
  }
  audio->free();
  printf("\n");
  exit(EXIT_SUCCESS);
}
