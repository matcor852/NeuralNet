/*
   Copyright (C) 2023 Matthieu Correia <matcor852@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

*/

#include "functions/cost.h"

const struct cost *get_cost(const char *restrict name) {
	for(size_t i = 0; i < (sizeof(costs) / sizeof(costs[0])); ++i)
		if(!strcmp(costs[i].name, name)) return &costs[i];
	errx(2, "get_cost: unknown cost function '%s'", name);
}
