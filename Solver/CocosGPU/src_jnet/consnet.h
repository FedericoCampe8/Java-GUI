/*********************************************************
  consnet.h
  --------------------------------------------------------
  generated at Mon Jul  5 11:20:30 1999
  by snns2c ( Bernward Kett 1995 ) 
*********************************************************/

extern int consnet(float *in, float *out, int init);

static struct {
  int NoOfInput;    /* Number of Input Units  */
  int NoOfOutput;   /* Number of Output Units */
  int(* propFunc)(float *, float*, int);
} consnetREC = {204,3,consnet};
