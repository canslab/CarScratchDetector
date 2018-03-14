#pragma once
#include <Windows.h>
class Timer
{
private:
	LARGE_INTEGER m_CPUFreq;
	LARGE_INTEGER m_startCPUClock;
	LARGE_INTEGER m_endCPUClock;
	bool m_bStarted;

public:
	void StartStopWatch();
	double EndStopWatch();

public:
	Timer();
	~Timer();
};