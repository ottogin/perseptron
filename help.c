#pragma once
#include <stdlib.h>

static float f_bin_pol(float a)
	{
		return 2 / (1 + expf(-a)) - 1;
	}

static float f_bin(float a)
	{
		return 1 / (1 + expf(-a));
	}
