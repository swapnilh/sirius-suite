inline	 int cons1(   __global char *stemmer_b, int i) {
		switch (stemmer_b[i]) {
			case 'a':
			case 'e':
			case 'i':
			case 'o':
			case 'u':
				return 0;
			default:
				return 1;
		}
	}

inline	   int cons(   __global char *stemmer_b, int i) {
		switch (stemmer_b[i]) {
			case 'a':
			case 'e':
			case 'i':
			case 'o':
			case 'u':
				return 0;
			case 'y':
				return (i == 0) ? 1 : !cons1(stemmer_b, i - 1);
			default:
				return 1;
		}
	}

inline	   int m(   __global char *stemmer_b, __global int *stemmer_j) {
		int n = 0;
		int i = 0;
		int j = *stemmer_j;
		while (1) {
			if (i > j) return n;
			if (!cons(stemmer_b, i)) break;
			i++;
		}
		i++;
		while (1) {
			while (1) {
				if (i > j) return n;
				if (cons(stemmer_b, i)) break;
				i++;
			}
			i++;
			n++;
			while (1) {
				if (i > j) return n;
				if (!cons(stemmer_b, i)) break;
				i++;
			}
			i++;
		}
	}


inline	   int vowelinstem(   __global char *stemmer_b, __global int *stemmer_j) {
		int i;
		int j = *stemmer_j;
		for (i = 0; i <= j; i++)
			if (!cons(stemmer_b, i)) return 1;
		return 0;
	}

	   int doublec(   __global char *stemmer_b, int j) {
		if (j < 1) return 0;
		if (stemmer_b[j] != stemmer_b[j - 1]) return 0;
	return cons(stemmer_b, j);
}

inline   int cvc(   __global char *stemmer_b, int i) {
	if (i < 2 || !cons(stemmer_b, i) || cons(stemmer_b, i - 1) || !cons(stemmer_b, i - 2)) return 0;
	{
		int ch = stemmer_b[i];
		if (ch == 'w' || ch == 'x' || ch == 'y') return 0;
	}
	return 1;
}
/*
inline   int memcmp1(__global const char *buffer1,  __constant char *buffer2, int start, int count) {
	if (!count) return (0);
	int i = start;
	int j = 1;
	int itr = 0;  
//	while (--count && (int)buffer1[i] == (int)buffer2[j]) {
	for (; itr<count; itr++)	{
		if (buffer1[i] != buffer2[j]) break;
//		if (*buffer1 != 'a') break;
		i++;
		j++;
	}
	return buffer1[i] - buffer2[j];
}
*/

inline   int ends(__global char *stemmer_b, __global int *stemmer_j, __global int *stemmer_k, __constant  char *s) {
	int length = s[0];
	int k = *stemmer_k;
	if (s[length] != stemmer_b[k]) return 0; 
	if (length > k + 1) return 0;
//	if (memcmp1(stemmer_b, s , k - length +1, length) != 0) return 0;'
	int count = length;
	int i = k - length + 1;
	int j = 0;
	while (--count && stemmer_b[i] == s[j]) {
		i++;
		j++;
	}
	*stemmer_j = k - length;
	return 1;
}

inline void memmove1(__global char *dst, __constant char *src, int start, int count) {
	int i = start, j = 1;
		while (count--) {
			dst[i] = src[j];
			i++;
			j++;
		}
}

inline   void setto(   __global char *stemmer_b, __global int *stemmer_j, __global int *stemmer_k, __constant  char *s) {
	int length = s[0];
	int j = *stemmer_j;
	memmove1(stemmer_b , s , j + 1, length);
	*stemmer_k = j + length;
}


inline   void r(   __global char *stemmer_b, __global int *stemmer_j, __global int *stemmer_k, __constant  char *s) {
	if (m(stemmer_b,stemmer_j) > 0) setto(stemmer_b,stemmer_j, stemmer_k, s);
}
      void step1ab(   __global char *stemmer_b, __global int *stemmer_j, __global int *stemmer_k ) {
   if (stemmer_b[*stemmer_k] == 's') {
   if (ends(stemmer_b, stemmer_j, stemmer_k,  "\04"   "sses"))
   *stemmer_k -= 2;
   else if (ends(stemmer_b, stemmer_j, stemmer_k,   "\03"   "ies"))
   setto(stemmer_b, stemmer_j, stemmer_k,   "\01"   "i");
   else if (stemmer_b[*stemmer_k - 1] != 's')
   *stemmer_k =(*stemmer_k)-1;
   }
   if (ends(stemmer_b, stemmer_j, stemmer_k,   "\03"   "eed")) {
   if (m(stemmer_b, stemmer_j) > 0) *stemmer_k = (*stemmer_k)-1;
   } else if ((ends(stemmer_b, stemmer_j, stemmer_k,   "\02"   "ed") ||
   ends(stemmer_b, stemmer_j, stemmer_k,   "\03"   "ing")) &&
   vowelinstem(stemmer_b, stemmer_j)) {
   *stemmer_k = *stemmer_j;
   if (ends(stemmer_b, stemmer_j, stemmer_k,
   "\02"
   "at"))
   setto(stemmer_b, stemmer_j, stemmer_k,
   "\03"
   "ate");
   else if (ends(stemmer_b, stemmer_j, stemmer_k,
   "\02"
   "bl"))
   setto(stemmer_b, stemmer_j, stemmer_k,
   "\03"
   "ble");
   else if (ends(stemmer_b, stemmer_j, stemmer_k,
   "\02"
   "iz"))
   setto(stemmer_b, stemmer_j, stemmer_k,
   "\03"
   "ize");
   else if (doublec(stemmer_b, *stemmer_k)) {
   *stemmer_k=(*stemmer_k)-1;
   {
   int ch = stemmer_b[*stemmer_k];
   if (ch == 'l' || ch == 's' || ch == 'z') *stemmer_k=(*stemmer_k)+1;
   }
   } else if (m(stemmer_b, stemmer_j) == 1 && cvc(stemmer_b, *stemmer_k))
   setto(stemmer_b, stemmer_j, stemmer_k,
   "\01"
   "e");
   }
   }


      void step1c(__global char *stemmer_b,  __global int *stemmer_j, __global int *stemmer_k) {
   if (ends(stemmer_b, stemmer_j, stemmer_k,
   "\01"
   "y") &&
   vowelinstem(stemmer_b, stemmer_j))
   stemmer_b[*stemmer_k] = 'i';
   }

      void step2(   __global char *stemmer_b,  __global int *stemmer_j, __global int *stemmer_k) {
   switch (stemmer_b[*stemmer_k - 1]) {
case 'a':
if (ends(stemmer_b, stemmer_j, stemmer_k,
			"\07"
			"ational")) {
	r(stemmer_b, stemmer_j, stemmer_k,
			"\03"
			"ate");
	break;
}
if (ends(stemmer_b, stemmer_j, stemmer_k,
			"\06"
			"tional")) {
	r(stemmer_b, stemmer_j, stemmer_k,
			"\04"
			"tion");
	break;
}
break;
case 'c':
if (ends(stemmer_b, stemmer_j, stemmer_k,
			"\04"
			"enci")) {
	r(stemmer_b, stemmer_j, stemmer_k,
			"\04"
			"ence");
	break;
}
if (ends(stemmer_b, stemmer_j, stemmer_k,
			"\04"
			"anci")) {
	r(stemmer_b, stemmer_j, stemmer_k,
			"\04"
			"ance");
	break;
}
break;
case 'e':
if (ends(stemmer_b, stemmer_j, stemmer_k,
			"\04"
			"izer")) {
	r(stemmer_b, stemmer_j, stemmer_k,
			"\03"
			"ize");
	break;
}
break;
case 'l':
if (ends(stemmer_b, stemmer_j, stemmer_k,
			"\03"
			"bli")) {
	r(stemmer_b, stemmer_j, stemmer_k,
			"\03"
			"ble");
	break;
}


if (ends(stemmer_b, stemmer_j, stemmer_k,
			"\04"
			"alli")) {
	r(stemmer_b, stemmer_j, stemmer_k,
			"\02"
			"al");
	break;
}
if (ends(stemmer_b, stemmer_j, stemmer_k,
			"\05"
			"entli")) {
	r(stemmer_b, stemmer_j, stemmer_k,
			"\03"
			"ent");
	break;
}
if (ends(stemmer_b, stemmer_j, stemmer_k,
			"\03"
			"eli")) {
	r(stemmer_b, stemmer_j, stemmer_k,
			"\01"
			"e");
	break;
}
if (ends(stemmer_b, stemmer_j, stemmer_k,
			"\05"
			"ousli")) {
	r(stemmer_b, stemmer_j, stemmer_k,
			"\03"
			"ous");
	break;
}
break;
case 'o':
if (ends(stemmer_b, stemmer_j, stemmer_k,
			"\07"
			"ization")) {
	r(stemmer_b, stemmer_j, stemmer_k,
			"\03"
			"ize");
	break;
}
if (ends(stemmer_b, stemmer_j, stemmer_k,
			"\05"
			"ation")) {
	r(stemmer_b, stemmer_j, stemmer_k,
			"\03"
			"ate");
	break;
}
if (ends(stemmer_b, stemmer_j, stemmer_k,
			"\04"
			"ator")) {
	r(stemmer_b, stemmer_j, stemmer_k,
			"\03"
			"ate");
	break;
}
break;
case 's':
if (ends(stemmer_b, stemmer_j, stemmer_k,
			"\05"
			"alism")) {
	r(stemmer_b, stemmer_j, stemmer_k,
			"\02"
			"al");
	break;
}
if (ends(stemmer_b, stemmer_j, stemmer_k,
			"\07"
			"iveness")) {
	r(stemmer_b, stemmer_j, stemmer_k,
			"\03"
			"ive");
	break;
}
if (ends(stemmer_b, stemmer_j, stemmer_k,
			"\07"
			"fulness")) {
	r(stemmer_b, stemmer_j, stemmer_k,
			"\03"
			"ful");
	break;
}
if (ends(stemmer_b, stemmer_j, stemmer_k,
			"\07"
			"ousness")) {
	r(stemmer_b, stemmer_j, stemmer_k,
			"\03"
			"ous");
	break;
}
break;
case 't':
if (ends(stemmer_b, stemmer_j, stemmer_k,
			"\05"
			"aliti")) {
	r(stemmer_b, stemmer_j, stemmer_k,
			"\02"
			"al");
	break;
}
if (ends(stemmer_b, stemmer_j, stemmer_k,
			"\05"
			"iviti")) {
	r(stemmer_b, stemmer_j, stemmer_k,
			"\03"
			"ive");
	break;
}
if (ends(stemmer_b, stemmer_j, stemmer_k,
			"\06"
			"biliti")) {
	r(stemmer_b, stemmer_j, stemmer_k,
			"\03"
			"ble");
	break;
}
break;
case 'g':
if (ends(stemmer_b, stemmer_j, stemmer_k,
			"\04"
			"logi")) {
	r(stemmer_b, stemmer_j, stemmer_k,
			"\03"
			"log");
	break;
}

}
}

   void step3(   __global char *stemmer_b,  __global int *stemmer_j, __global int *stemmer_k) {
	switch (stemmer_b[*stemmer_k]) {
		case 'e':
			if (ends(stemmer_b, stemmer_j, stemmer_k,
						"\05"
						"icate")) {
				r(stemmer_b, stemmer_j, stemmer_k,
						"\02"
						"ic");
				break;
			}
			if (ends(stemmer_b, stemmer_j, stemmer_k,
						"\05"
						"ative")) {
				r(stemmer_b, stemmer_j, stemmer_k,
						"\00"
						"");
				break;
			}
			if (ends(stemmer_b, stemmer_j, stemmer_k,
						"\05"
						"alize")) {
				r(stemmer_b, stemmer_j, stemmer_k,
						"\02"
						"al");
				break;
			}
			break;
		case 'i':
			if (ends(stemmer_b, stemmer_j, stemmer_k,
						"\05"
						"iciti")) {
				r(stemmer_b, stemmer_j, stemmer_k,
						"\02"
						"ic");
				break;
			}
			break;
		case 'l':
			if (ends(stemmer_b, stemmer_j, stemmer_k,
						"\04"
						"ical")) {
				r(stemmer_b, stemmer_j, stemmer_k,
						"\02"
						"ic");
				break;
			}
			if (ends(stemmer_b, stemmer_j, stemmer_k,
						"\03"
						"ful")) {
				r(stemmer_b, stemmer_j, stemmer_k,
						"\00"
						"");
				break;
			}
			break;
		case 's':
			if (ends(stemmer_b, stemmer_j, stemmer_k,
						"\04"
						"ness")) {
				r(stemmer_b, stemmer_j, stemmer_k,
						"\00"
						"");
				break;
			}
			break;
	}
}


   void step4(   __global char *stemmer_b,  __global int *stemmer_j, __global int *stemmer_k) {
	switch (stemmer_b[*stemmer_k - 1]) {
		case 'a':
			if (ends(stemmer_b, stemmer_j, stemmer_k,
						"\02"
						"al"))
				break;
			return;
		case 'c':
			if (ends(stemmer_b, stemmer_j, stemmer_k,
						"\04"
						"ance"))
				break;
			if (ends(stemmer_b, stemmer_j, stemmer_k,
						"\04"
						"ence"))
				break;
			return;
		case 'e':
			if (ends(stemmer_b, stemmer_j, stemmer_k,
						"\02"
						"er"))
				break;
			return;
		case 'i':
			if (ends(stemmer_b, stemmer_j, stemmer_k,
						"\02"
						"ic"))
				break;
			return;
		case 'l':
			if (ends(stemmer_b, stemmer_j, stemmer_k,
						"\04"
						"able"))
				break;
			if (ends(stemmer_b, stemmer_j, stemmer_k,
						"\04"
						"ible"))
				break;
			return;
		case 'n':
			if (ends(stemmer_b, stemmer_j, stemmer_k,
						"\03"
						"ant"))
				break;
			if (ends(stemmer_b, stemmer_j, stemmer_k,
						"\05"
						"ement"))
				break;
			if (ends(stemmer_b, stemmer_j, stemmer_k,
						"\04"
						"ment"))
				break;
			if (ends(stemmer_b, stemmer_j, stemmer_k,
						"\03"
						"ent"))
				break;
			return;
		case 'o':
			if (ends(stemmer_b, stemmer_j, stemmer_k,
						"\03"
						"ion") &&
					(stemmer_b[*stemmer_j] == 's' || stemmer_b[*stemmer_j] == 't'))
				break;
			if (ends(stemmer_b, stemmer_j, stemmer_k,
						"\02"
						"ou"))
				break;
			return;
		case 's':
			if (ends(stemmer_b, stemmer_j, stemmer_k,
						"\03"
						"ism"))
				break;
			return;
		case 't':
			if (ends(stemmer_b, stemmer_j, stemmer_k,
						"\03"
						"ate"))
				break;
			if (ends(stemmer_b, stemmer_j, stemmer_k,
						"\03"
						"iti"))
				break;
			return;
		case 'u':
			if (ends(stemmer_b, stemmer_j, stemmer_k,
						"\03"
						"ous"))
				break;
			return;
		case 'v':
			if (ends(stemmer_b, stemmer_j, stemmer_k,
						"\03"
						"ive"))
				break;
			return;
		case 'z':
			if (ends(stemmer_b, stemmer_j, stemmer_k,
						"\03"
						"ize"))
				break;
			return;
		default:
			return;
	}
	if (m(stemmer_b, stemmer_j) > 1) *stemmer_k = *stemmer_j;
}


 void step5(__global char *stemmer_b,  __global int *stemmer_j, __global int *stemmer_k) {
	*stemmer_j = *stemmer_k;
	if (stemmer_b[*stemmer_k] == 'e') {
		int a = m(stemmer_b, stemmer_j);
		if ((a > 1 || a == 1) && !(cvc(stemmer_b, *stemmer_k - 1))) 
		*stemmer_k=(*stemmer_k)-1;
	}
	if (stemmer_b[*stemmer_k] == 'l' && doublec(stemmer_b, *stemmer_k) && m(stemmer_b, stemmer_j) > 1) *stemmer_k=(*stemmer_k)-1;
}

__kernel void stem_gpu(__global char *stemmer_b,  __global int *stemmer_j, __global int *stemmer_k, int words) {
	int tid = get_group_id(0) * get_local_size(0) + get_local_id(0);
	if (tid < words) {
		if (stemmer_k[tid] <= 1) {
			return;
		}
		
		   step1ab(&stemmer_b[tid*(32+1)], &(stemmer_j[tid]), &(stemmer_k[tid]));
//		   step1c(&stemmer_b[tid*(32+1)], &(stemmer_j[tid]), &(stemmer_k[tid]));
//		   step2(&stemmer_b[tid*(32+1)], &(stemmer_j[tid]), &(stemmer_k[tid]));
//		   step3(&stemmer_b[tid*(32+1)], &(stemmer_j[tid]), &(stemmer_k[tid]));
//		   step4(&stemmer_b[tid*(32+1)], &(stemmer_j[tid]), &(stemmer_k[tid]));
			
//		   step5(&stemmer_b[tid*(32+1)], &(stemmer_j[tid]), &(stemmer_k[tid])); 
		   stemmer_b[tid*(32+1)+stemmer_k[tid] + 1] = 0;
	}
}
