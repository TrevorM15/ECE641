
#include <math.h>
#include "tiff.h"
#include "allocate.h"
#include "randlib.h"
#include "typeutil.h"

void error(char *name);

int main (int argc, char **argv) 
{
  FILE *fp;
  struct TIFF_img input_img, output_img, output_err_img, output_blur_img, rest_img, rest_blur_img;
  double **img, **error_img, **blur_img,**clean_img,**cb_img,**bnn_img, sigma_hat[20],cost[20], p=0,N, gcx=0, gcx2=0,v,theta1=0, theta2=0,**e, H[5][5];
  int32_t i,j,k,l,m,pixel;
  double sigma = pow(16,2);

  double g[3][3] = {
    {1.0/12, 1.0/6, 1.0/12},
    {1.0/6, 0, 1.0/6},
    {1.0/12, 1.0/6,1.0/12}
  };

  double h[5][5] = {
    { 1.0/81,2.0/81,3.0/81,2.0/81,1.0/81},
    {2.0/81,4.0/81,6.0/81,4.0/81,2.0/81},
    {3.0/81,6.0/81,0,6.0/81,4.0/81},
    {2.0/81,4.0/81,6.0/81,4.0/81,2.0/81},
    { 1.0/81,2.0/81,3.0/81,2.0/81,1.0/81},
  };

  for (k = 0; k < 5; k++)
    for(l = 0; l < 5; l++){
      H[k][l]=h[k][l];
}
H[2][2]=9.0/81;
  
  if ( argc != 2 ) error( argv[0] );

  /* open image file */
  if ( ( fp = fopen ( argv[1], "rb" ) ) == NULL ) {
    fprintf ( stderr, "cannot open file %s\n", argv[1] );
    exit ( 1 );
  }

  /* read image */
  if ( read_TIFF ( fp, &input_img ) ) {
    fprintf ( stderr, "error reading file %s\n", argv[1] );
    exit ( 1 );
  }

  /* close image file */
  fclose ( fp );

  /* check the type of image data */
  if ( input_img.TIFF_type != 'g' ) {
    fprintf ( stderr, "error:  image must be grayscale\n" );
    exit ( 1 );
  }

  /* Allocate image of double precision floats */
  img = (double **)get_img(input_img.width,input_img.height,sizeof(double));
  blur_img = (double **)get_img(input_img.width,input_img.height,sizeof(double));
  error_img = (double **)get_img(input_img.width,input_img.height,sizeof(double));
  clean_img = (double **)get_img(input_img.width,input_img.height,sizeof(double));
  cb_img = (double **)get_img(input_img.width,input_img.height,sizeof(double));
  bnn_img = (double **)get_img(input_img.width,input_img.height,sizeof(double));
  e = (double **)get_img(input_img.width,input_img.height,sizeof(double));
  N = input_img.width*input_img.height;

  
    /* copy input image to double array */
  for ( i = 0; i < input_img.height; i++ )
  for ( j = 0; j < input_img.width; j++ ) {
    img[i][j] = input_img.mono[i][j];
    error_img[i][j] = 0;
    blur_img[i][j] = img[i][j];
    clean_img[i][j]=0;
    bnn_img[i][j] = img[i][j];
    e[i][j]=0;
 }
  


  
    /* calculate non-causal prediction error */

  for ( i = 1; i < input_img.height-1; i++ ){
  for ( j = 1; j < input_img.width-1; j++ ) {
    for ( k = 0; k < 3; k++){
      for (l = 0; l < 3; l++){
	gcx += g[k][l]*img[i+k-1][j+l-1]; 
      }
    }
    error_img[i][j] = img[i][j]-gcx+127;
    gcx=0;

    }
  }


  /*compute sigma_ML in the range 0.1<=p<=2 and plot sigma_ML vs p */
 for ( m = 1; m < 21; m++){
 for ( i = 1; i < input_img.height-1; i++ ){
  for ( j = 1; j < input_img.width-1; j++ ) {
    for ( k = 0; k < 3; k++){
      for (l = 0; l < 3; l++){
	gcx += g[k][l]*pow(abs(img[i+k-1][j+l-1]-img[i][j]) ,0.1*m); 
      }
    }
  }
 }

 sigma_hat[m]= gcx/N; 
 //printf("%f \n ", sigma_hat[m]);
  }

  
  
  /* Set seed for random noise generator */
  srandom2(1);

  /* Add noise to image */
  
  for ( i = 1; i < input_img.height-1; i++ )
  for ( j = 1; j < input_img.width-1; j++ ) {
    img[i][j] += sigma*normal();
    clean_img[i][j] = img[i][j];
  }
   

  /* compute MAP using ICD*/

  for (m = 0; m < 20; m++){
  for ( i = 1; i < input_img.height-1; i++ ){
  for ( j = 1; j < input_img.width-1; j++ ) {
     for ( k = 0; k < 3; k++){
      for (l = 0; l < 3; l++){
	gcx += g[k][l]*clean_img[i+k-1][j+l-1];
	gcx2 += g[k][l]*pow(clean_img[i+k-1][j+l-1]-clean_img[i][j],2);
	
      }
    }
     clean_img[i][j] = (img[i][j]+sigma/(1*sigma_hat[20])*gcx)/(1+sigma/(1*sigma_hat[20]));
    gcx=0;
    if (clean_img[i][j] <0){clean_img[i][j]=0;}
    cost[m] += (1/(2*sigma))*pow(img[i][j]-clean_img[i][j],2)+(1/(2*sigma_hat[20]))*gcx2;
    gcx2=0;
  }
  }
  //printf("iteration %d: %f \n ", m, cost[m]);
  }

  

  /* make blurry image */

  for ( i = 2; i < input_img.height-2; i++ ){
  for ( j = 2; j < input_img.width-2; j++ ) {
    blur_img[i][j] = 9.0/81*blur_img[i][j];
    for ( k = 0; k < 5; k++){
      for (l = 0; l < 5; l++){
       blur_img[i][j] += h[k][l]*blur_img[i+k-2][j+l-2];
       
      }
    }
    bnn_img[i][j] = blur_img[i][j];
  }}
  
  /* Add noise to image */
  for ( i = 1; i < input_img.height-1; i++ )
  for ( j = 1; j < input_img.width-1; j++ ) {
   blur_img[i][j] += 16*normal();
   if(blur_img[i][j]>255) {
      blur_img[i][j] = 255;
    }
    else {
      if(blur_img[i][j]<0) blur_img[i][j] = 0;
      else blur_img[i][j] = blur_img[i][j];
    }
   cb_img[i][j]=blur_img[i][j];
   e[i][j] = blur_img[i][j]-bnn_img[i][j];
  }

  
  /* TODO: compute ICD */
  
  for (m = 0; m <20; m++)
 for ( i = 2; i < input_img.height-2; i++ )
  for ( j = 2; j < input_img.width-2; j++ ) {
    v = cb_img[i][j];
    for ( k = 0; k < 5; k++){
      for (l = 0; l < 5; l++){
	theta1-= H[k][l]*e[i+k-2][j+l-2]/16;
	theta2 += pow(H[k][l],2)/16;
      }
    }
    for ( k = 0; k < 3; k++){
      for (l = 0; l < 3; l++){
       gcx += g[k][l]*blur_img[i+k-1][j+l-1];
      }
    }
    cb_img[i][j] = (theta2*v-theta1+(1.0/sigma_hat[10])*gcx)/(theta2+ (1.0/sigma_hat[10]));
    if( cb_img[i][j] <0){cb_img[i][j]=0;}
    for ( k = 0; k < 5; k++){
      for (l = 0; l < 5; l++){
	e[i+k-2][j+l-2] = e[i+k-2][j+l-2]-H[k][l]*(cb_img[i][j]-v);
      }
    }

    gcx=0;
    theta1=0;
    theta2=0;
  }
  
  

  /* set up structure for output image */
  /* to allocate a full color image use type 'c' */
  get_TIFF ( &output_err_img, input_img.height, input_img.width, 'g' );
   get_TIFF ( &output_img, input_img.height, input_img.width, 'g' );
   get_TIFF ( &rest_img, input_img.height, input_img.width, 'g' );
   get_TIFF ( &rest_blur_img, input_img.height, input_img.width, 'g' );
  get_TIFF ( &output_blur_img, input_img.height, input_img.width, 'g' );
  //1
  /* copy error img to output image */
  for ( i = 0; i < input_img.height; i++ )
  for ( j = 0; j < input_img.width; j++ ) {
    pixel = (int32_t)error_img[i][j];
    if(pixel>255) {
      output_err_img.mono[i][j] = 255;
    }
    else {
      if(pixel<0) output_err_img.mono[i][j] = 0;
      else output_err_img.mono[i][j] = pixel;
    }
  }

  /* open image file */
  if ( ( fp = fopen ( "output_err_img.tif", "wb" ) ) == NULL ) {
    fprintf ( stderr, "cannot open file output_img.tif\n");
    exit ( 1 );
  }

  /* write image */
  if ( write_TIFF ( fp, &output_err_img ) ) {
    fprintf ( stderr, "error writing TIFF file %s\n", argv[2] );
    exit ( 1 );
  }

  /* close image file */
  fclose ( fp );
  // 2
    /* copy noisy img to output image */
  for ( i = 0; i < input_img.height; i++ )
  for ( j = 0; j < input_img.width; j++ ) {
    pixel = (int32_t)img[i][j];
    if(pixel>255) {
      output_img.mono[i][j] = 255;
    }
    else {
      if(pixel<0) output_img.mono[i][j] = 0;
      else output_img.mono[i][j] = pixel;
    }
  }

  /* open image file */
  if ( ( fp = fopen ( "noisy_img.tif", "wb" ) ) == NULL ) {
    fprintf ( stderr, "cannot open file noisy_img.tif\n");
    exit ( 1 );
  }

  /* write image */
  if ( write_TIFF ( fp, &output_img ) ) {
    fprintf ( stderr, "error writing TIFF file %s\n", argv[2] );
    exit ( 1 );
  }

  /* close image file */
  fclose ( fp );
  // 3
  /* copy  noisy blurry img to output image */
  for ( i = 0; i < input_img.height; i++ )
  for ( j = 0; j < input_img.width; j++ ) {
    pixel = (int32_t)blur_img[i][j];
    if(pixel>255) {
      output_blur_img.mono[i][j] = 255;
    }
    else {
      if(pixel<0) output_blur_img.mono[i][j] = 0;
      else output_blur_img.mono[i][j] = pixel;
    }
  }

  /* open image file */
  if ( ( fp = fopen ( "noisy_blurred_img.tif", "wb" ) ) == NULL ) {
    fprintf ( stderr, "cannot open file noisy_blurred_img.tif\n");
    exit ( 1 );
  }

  /* write image */
  if ( write_TIFF ( fp, &output_blur_img ) ) {
    fprintf ( stderr, "error writing TIFF file %s\n", argv[2] );
    exit ( 1 );
  }

  /* close image file */
  fclose ( fp );

  // 4
    /* copy restored noisy image to output image */
  for ( i = 0; i < input_img.height; i++ )
  for ( j = 0; j < input_img.width; j++ ) {
    pixel = (int32_t)clean_img[i][j];
    if(pixel>255) {
      rest_img.mono[i][j] = 255;
    }
    else {
      if(pixel<0) rest_img.mono[i][j] = 0;
      else rest_img.mono[i][j] = pixel;
    }
  }

  /* open image file */
  if ( ( fp = fopen ( "restored_img.tif", "wb" ) ) == NULL ) {
    fprintf ( stderr, "cannot open file restored_img.tif\n");
    exit ( 1 );
  }

  /* write image */
  if ( write_TIFF ( fp, &rest_img ) ) {
    fprintf ( stderr, "error writing TIFF file %s\n", argv[2] );
    exit ( 1 );
  }

  /* close image file */
  fclose ( fp );

  // 5

    /* copy restored noisy blurred image to output image */
  for ( i = 0; i < input_img.height; i++ )
  for ( j = 0; j < input_img.width; j++ ) {
    pixel = (int32_t)cb_img[i][j];
    if(pixel>255) {
      rest_blur_img.mono[i][j] = 255;
    }
    else {
      if(pixel<0) rest_blur_img.mono[i][j] = 0;
      else rest_blur_img.mono[i][j] = pixel;
    }
  }

  /* open image file */
  if ( ( fp = fopen ( "restored_blurred_img.tif", "wb" ) ) == NULL ) {
    fprintf ( stderr, "cannot open file restored_blurred_img.tif\n");
    exit ( 1 );
  }

  /* write image */
  if ( write_TIFF ( fp, &rest_blur_img ) ) {
    fprintf ( stderr, "error writing TIFF file %s\n", argv[2] );
    exit ( 1 );
  }

  /* close image file */
  fclose ( fp );


  
  /* de-allocate space which was used for the images */
  free_TIFF ( &(input_img) );
  free_TIFF ( &(output_img) );
  free_TIFF ( &(output_err_img) );
  free_TIFF ( &(output_blur_img) );
  free_TIFF ( &(rest_img));
  free_TIFF ( &(rest_blur_img));
  
  free_img( (void**)img);
  free_img( (void**)error_img);
  free_img( (void**)blur_img);
  free_img ( (void**)clean_img);
  free_img ( (void**)bnn_img);
  free_img ( (void**)cb_img);


  return(0);
}

void error(char *name)
{
    printf("usage:  %s  image.tiff \n\n",name);
    printf("this program reads in a grayscale TIFF image.\n");
    printf("adds noise,then MAP estimation is performed\n");
    printf("and writes out the result as an 8-bit image\n");
    printf("with the name 'output_img.tiff'.\n");
    exit(1);
}
