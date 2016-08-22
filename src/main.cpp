#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <IL/il.h>
#include <IL/ilu.h>
#include <IL/ilut.h> 
#include <pthread.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <iostream>
#include <list>
#include <sstream>
#include <string>
#include <stdexcept>
/*
#ifndef WIN32 //if using windows then do windows specific stuff.
#define WIN32_LEAN_AND_MEAN //remove MFC overhead from windows.h which can cause slowness
#define WIN32_EXTRA_LEAN

#include <windows.h>
#endif
*/
#include <GL/gl.h>
#include <GL/glext.h>
#include <GL/glfw3.h>

struct PIDCoefficient
{
	float p;
	float i;
	float d;
};

struct PID
{
	float v;
	float p;
	float i;
	float d;
	
	void update(float newValue, const PIDCoefficient &coefficient)
	{
		float error = newValue - v;
		d = error - p;
		i += error;
		p = error;
		v += p * coefficient.p + i * coefficient.i + d * coefficient.d;
		if (v < 0)
		{
			v = p = i = d = 0;
		}
	}
	
	PID &operator =(float newValue)
	{
		v = newValue;
		p = i = d = 0;
		return *this;
	}
	
	operator float()
	{
		return v;
	}
};

struct Face
{
	PID x;
	PID y;
	PID w;
	PID h;
	int age;
	GLuint image;
	bool mirrorx, mirrory;
	int noiseTimeW, noiseTimeH;
	float noiseW, noiseH;
};

const char *cascadeName = "data/haarcascade_frontalface_default.xml";
const float scaleFactor = 0.5f;
const cv::Size minSize(50, 50);
const int maxFaces = 20;
const int maxAge = 5;
const int limitAge = -20;
const int captureWidth = 1920;
const int captureHeight = 1080;
const int screenWidth = 1920;
const int screenHeight = 1080;
const float maxDistance = 200.f;
const float faceFactor = 1.3f;
const float noiseFactor = 20.f;
const int noiseMinTimeout = 10;
const int noiseMaxTimeout = 30;
const PIDCoefficient positionPID = {0.9f, 0.1f, 0.05f};
const PIDCoefficient sizePID = {0.2f, 0.2f, 0.1f};

const char *imageNames[] = {
	"data/anonymous.png",
	"data/awesome-smiley.png",
	"data/derpy-hooves.png",
	"data/dolan.png",
	"data/forever-alone.png",
	"data/i-see.png",
	"data/me-gusta.png",
	"data/nyan-cat.png",
	"data/pedobear.png",
	"data/rage.png",
	"data/rainbow-dash.png",
	"data/trollface.png",
	"data/twilight-sparkle.png",
	"data/yao-ming.png",
	NULL
};

cv::CascadeClassifier cascade;
cv::Mat frame, gray, resized;
std::vector<cv::Rect> rects;
Face faces[maxFaces];
cv::VideoCapture capture;

GLuint captureTexture;
std::list<GLuint> imageTextures;

cv::Mat screenshot(screenHeight, screenWidth, CV_8UC3);

pthread_t thread;
pthread_mutex_t imageMutex;

void *recognition(void *)
{
	for (;;)
	{
		if (gray.cols > 0)
		{
			pthread_mutex_lock(&imageMutex);
			cv::Size sz(cvRound(gray.cols * scaleFactor), cvRound(gray.rows * scaleFactor));
			cv::resize(gray, resized, sz);
			pthread_mutex_unlock(&imageMutex);
			
			cv::transpose(resized, resized);
			cv::flip(resized, resized, 0);
			
			cascade.detectMultiScale(
				resized,
				rects,
				1.2,
				3,
				CV_HAAR_SCALE_IMAGE,
				minSize
			);
		}
	}
	
	return NULL;
}

void keyCallback(GLFWwindow window, int key, int action)
{
	if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
	{
		time_t now;
		time(&now);
		char *datetime = ctime(&now);
		for (int i = 0, n = strlen(datetime); i < n; ++i)
		{
			if (datetime[i] == ' ' || datetime[i] == ':')
				datetime[i] = '_';
			else if (datetime[i] == '\n')
				datetime[i] = '\0';
		}
		std::ostringstream oss;
		oss << "screenshots/" << datetime << ".jpg";
		std::string filename(oss.str());
		glReadPixels(0, 0, screenWidth, screenHeight, GL_BGR, GL_UNSIGNED_BYTE, screenshot.data);
		cv::imwrite(filename, screenshot);
	}
}

GLuint nextImage()
{
	int index = rand() % (imageTextures.size() / 2);
	std::list<GLuint>::iterator it = imageTextures.begin();
	for (int i = 0; i < index; ++i, ++it)
	{
	}
	GLuint image = *it;
	imageTextures.erase(it);
	imageTextures.push_back(image);
	return image;
}

int main(int argc, char **argv)
{
	srand(time(NULL));
	
	for (int i = 0; i < maxFaces; ++i)
	{
		faces[i].age = limitAge;
		faces[i].noiseTimeW = 0;
		faces[i].noiseTimeH = 0;
	}
	
	if (!cascade.load(cascadeName))
		return std::cerr << "ERROR: Could not load cascade classifier \"" << cascadeName << "\"." << std::endl, -1;
	
	capture.open(-1);
	if (!capture.isOpened())
		return std::cerr << "ERROR: Could not access to camera." << std::endl, -1;
	
	capture.set(CV_CAP_PROP_FRAME_WIDTH, captureWidth);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, captureHeight);
	
	glfwInit();

	glfwOpenWindowHint(GLFW_WINDOW_RESIZABLE, false);
	glfwOpenWindowHint(GLFW_FSAA_SAMPLES, 4);
	glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 3);
	glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 3);
	glfwOpenWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

	GLFWwindow window = glfwOpenWindow(screenWidth, screenHeight, GLFW_FULLSCREEN, "Narkissos", NULL);
	glfwSetInputMode(window, GLFW_CURSOR_MODE, GLFW_CURSOR_HIDDEN);
	glfwSetKeyCallback(&keyCallback);
	
	glMatrixMode(GL_PROJECTION);
	glOrtho(0, screenWidth, screenHeight, 0, -1.0, 1.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	
	glDisable(GL_SMOOTH);
	glDisable(GL_LIGHTING);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
	glCullFace(GL_BACK);
    glEnable(GL_CULL_FACE);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	
	glPixelStorei(GL_PACK_ALIGNMENT, (screenshot.step & 3) ? 1 : 4);
	glPixelStorei(GL_PACK_ROW_LENGTH, screenshot.step / screenshot.elemSize());
	
	ilInit();
	iluInit();
	ilutRenderer(ILUT_OPENGL);

	ilutEnable(ILUT_OPENGL_CONV);

	glGenTextures(1, &captureTexture);
	glBindTexture(GL_TEXTURE_2D, captureTexture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	
	for (const char **ptr = imageNames; *ptr; ++ptr)
	{
		GLuint texture = ilutGLLoadImage((char*)*ptr);
		imageTextures.push_back(texture);
		glBindTexture(GL_TEXTURE_2D, texture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	}
	
	for (int i = 0; i < 100; ++i)
	{
		nextImage();
	}
	
	pthread_mutex_init(&imageMutex, NULL);
	pthread_create(&thread, NULL, recognition, NULL);
	
	for (;;)
	{
		glfwPollEvents();

		if (glfwGetKey(window, GLFW_KEY_ESC) == GLFW_PRESS || !glfwIsWindow(window)) {
			break;
		}
		
		capture >> frame;
		if (frame.empty())
		{
			continue;
		}

		pthread_mutex_lock(&imageMutex);
		cvtColor(frame, gray, CV_BGR2GRAY);
		pthread_mutex_unlock(&imageMutex);
		
		int count = std::min((int)rects.size(), maxFaces);
		for (int rectIndex = 0; rectIndex < count; ++rectIndex)
		{
			cv::Rect &rect = rects[rectIndex];
			
			float w = rect.height * screenWidth / captureWidth / 2.f / scaleFactor;
			float h = rect.width * screenHeight / captureHeight / 2.f / scaleFactor;
			float x = rect.y * screenWidth / captureWidth / scaleFactor + w;
			float y = screenHeight - rect.x * screenHeight / captureHeight / scaleFactor - h;
			
			w *= faceFactor;
			h *= faceFactor;
			
			int faceIndex;
			for (faceIndex = 0; faceIndex < maxFaces; ++faceIndex)
			{
				Face &face = faces[faceIndex];
				if (fabs(x - face.x) + fabs(y - face.y) < maxDistance && face.age > limitAge)
				{
					face.x.update(x, positionPID);
					face.y.update(y, positionPID);
					face.w.update(w + face.noiseW, sizePID);
					face.h.update(h + face.noiseH, sizePID);
					
					face.age = maxAge;
					break;
				}
			}
			
			if (faceIndex == maxFaces)
			{
				for (faceIndex = 0; faceIndex < maxFaces; ++faceIndex)
				{
					Face &face = faces[faceIndex];
					if (face.age == limitAge)
					{
						face.x = x;
						face.y = y;
						face.w = 0;
						face.h = 0;
						
						face.image = nextImage();
						face.mirrorx = (rand() % 2 == 1);
						face.mirrory = false; //(rand() % 2 == 1);
						
						face.w.update(w, sizePID);
						face.h.update(h, sizePID);
						
						face.age = maxAge;
						break;
					}
				}
			}
		}
		
		glBindTexture(GL_TEXTURE_2D, captureTexture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, frame.cols, frame.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, frame.data);
		
		glBegin(GL_QUADS);
		glTexCoord2d(1, 1); glVertex2f(0.0, 0.0);
		glTexCoord2d(1, 0); glVertex2f(0.0, screenHeight);
		glTexCoord2d(0, 0); glVertex2f(screenWidth, screenHeight);
		glTexCoord2d(0, 1); glVertex2f(screenWidth, 0.0);
		glEnd();

		for (int faceIndex = 0; faceIndex < maxFaces; ++faceIndex)
		{
			Face &face = faces[faceIndex];
			
			if (face.age > limitAge)
			{
				--face.age;
			}
			
			if (face.age < 0)
			{
				face.w.update(0, sizePID);
				face.h.update(0, sizePID);
			}
			
			if (--face.noiseTimeW < 0)
			{
				face.noiseTimeW = noiseMinTimeout + (noiseMaxTimeout - noiseMinTimeout) * (rand() / (float)RAND_MAX);
				face.noiseW = noiseFactor * (rand() / (float)RAND_MAX * 2.f - 1.f);
			}
			
			if (--face.noiseTimeH < 0)
			{
				face.noiseTimeH = noiseMinTimeout + (noiseMaxTimeout - noiseMinTimeout) * (rand() / (float)RAND_MAX);
				face.noiseH = noiseFactor * (rand() / (float)RAND_MAX * 2.f - 1.f);
			}
			
			glBindTexture(GL_TEXTURE_2D, face.image);
			glBegin(GL_QUADS);
			glTexCoord2d(face.mirrorx ? 0 : 1, face.mirrory ? 0 : 1); glVertex2f(face.x - face.w, face.y - face.h);
			glTexCoord2d(face.mirrorx ? 1 : 0, face.mirrory ? 0 : 1); glVertex2f(face.x - face.w, face.y + face.h);
			glTexCoord2d(face.mirrorx ? 1 : 0, face.mirrory ? 1 : 0); glVertex2f(face.x + face.w, face.y + face.h);
			glTexCoord2d(face.mirrorx ? 0 : 1, face.mirrory ? 1 : 0); glVertex2f(face.x + face.w, face.y - face.h);
			glEnd();
		}
		
		glfwSwapBuffers();
	}
	
	glfwTerminate();
	
	return 0;
}