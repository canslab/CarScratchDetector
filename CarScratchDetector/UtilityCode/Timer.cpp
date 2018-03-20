#include "..\stdafx.h"
#include "Timer.h"

void Timer::StartStopWatch()
{
	m_bStarted = true;
	QueryPerformanceCounter(&m_startCPUClock);
}

double Timer::EndStopWatch()
{
	double elapsedTime = 0;
	if (m_bStarted == false)
	{
		elapsedTime = -1;
	}
	else
	{
		QueryPerformanceCounter(&m_endCPUClock);
		m_bStarted = false;
		elapsedTime = (double)(m_endCPUClock.QuadPart - m_startCPUClock.QuadPart) / m_CPUFreq.QuadPart;
	}
	// return the elapsed time in milliseconds.
	return elapsedTime * 1000;
}

Timer::Timer() :m_bStarted(false)
{
	LARGE_INTEGER tempFreq;
	if (QueryPerformanceFrequency(&tempFreq))
	{
		m_CPUFreq = tempFreq;
	}
}

Timer::~Timer()
{

}
