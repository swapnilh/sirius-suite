#define TRUE 1
#define FALSE 0 
#define INC 32 


struct stemmer {
	char b[INC + 1]; 
	int k;           
	int j;           
};


inline int cons1(   __global struct stemmer *z, int i) {
	switch (z->b[i]) {
		case 'a':
		case 'e':
		case 'i':
		case 'o':
		case 'u':
			return FALSE;
		default:
			return TRUE;
	}
}

 inline  int cons(   __global struct stemmer *z, int i) {
	switch (z->b[i]) {
		case 'a':
		case 'e':
		case 'i':
		case 'o':
		case 'u':
			return FALSE;
		case 'y':
			return (i == 0) ? TRUE : !cons1(z, i - 1);
		default:
			return TRUE;
	}
}

 inline  int m(   __global struct stemmer *z) {
	int n = 0;
	int i = 0;
	int j = z->j;
	while (TRUE) {
		if (i > j) return n;
		if (!cons(z, i)) break;
		i++;
	}
	i++;
	while (TRUE) {
		while (TRUE) {
			if (i > j) return n;
			if (cons(z, i)) break;
			i++;
		}
		i++;
		n++;
		while (TRUE) {
			if (i > j) return n;
			if (!cons(z, i)) break;
			i++;
		}
		i++;
	}
}


 inline  int vowelinstem(   __global struct stemmer *z) {
	int j = z->j;
	int i;
	for (i = 0; i <= j; i++)
		if (!cons(z, i)) return TRUE;
	return FALSE;
}

 inline  int doublec(   __global struct stemmer *z, int j) {
	if (j < 1) return FALSE;
	if (z->b[j] != z->b[j - 1]) return FALSE;
	return cons(z, j);
}

 inline  int cvc(   __global struct stemmer *z, int i) {
	if (i < 2 || !cons(z, i) || cons(z, i - 1) || !cons(z, i - 2)) return FALSE;
	{
		int ch = z->b[i];
		if (ch == 'w' || ch == 'x' || ch == 'y') return FALSE;
	}
	return TRUE;
}
inline int ends(__global struct stemmer *z, __constant char *s) {
	int length = s[0];
	int k = z->k;
	if (s[length] != z->b[k]) return FALSE; 
	if (length > k + 1) return FALSE;
	__local int count ;
	count = 1;
	int flag = 0;
	while (count  < length) {
		if (s[count] != z->b[k - length + count]) { 
			flag = 1;
		}
		count++;
	}
	if (flag == 0) {
		z->j = k - length;
		return TRUE;
	}
	else {
		return FALSE;
	}
}

void memmove1(__global char *dst, __constant char *src, int start, int count) {
	int i = start, j = 1;
	// Changing memmove1 to make it easier to write OpenCL code, and this works as there cannot be any overlap possible between the two strings
//	if (dst <= src || dst >= (src + count)) {
		while (count--) {
			dst[i] = src[j];
			i++;
			j++;
		}
	/*
	}
	else {
		dst_t = dst + count - 1;
		src_t = src + count - 1;
		while (count--) {
			*dst_t-- = *src_t--;
		}
	}*/
}

 inline  void setto(   __global struct stemmer *z,  __constant char *s) {
	int length = s[0];
	int j = z->j;
	memmove1(z->b , s , j + 1, length);
	z->k = j + length;
}


 inline  void r(   __global struct stemmer *z, __constant char *s) {
	if (m(z) > 0) setto(z, s);
}
 inline     void step1ab(   __global struct stemmer *z) {
   if (z->b[z->k] == 's') {
   if (ends(z,
   "\04"
   "sses"))
   z->k -= 2;
   else if (ends(z,
   "\03"
   "ies"))
   setto(z,
   "\01"
   "i");
   else if (z->b[z->k - 1] != 's')
   z->k--;
   }
   if (ends(z,
   "\03"
   "eed")) {
   if (m(z) > 0) z->k--;
   } else if ((ends(z,
   "\02"
   "ed") ||
   ends(z,
   "\03"
   "ing")) &&
   vowelinstem(z)) {
   z->k = z->j;
   if (ends(z,
   "\02"
   "at"))
   setto(z,
   "\03"
   "ate");
   else if (ends(z,
   "\02"
   "bl"))
   setto(z,
   "\03"
   "ble");
   else if (ends(z,
   "\02"
   "iz"))
   setto(z,
   "\03"
   "ize");
   else if (doublec(z, z->k)) {
   z->k--;
   {
   int ch = z->b[z->k];
   if (ch == 'l' || ch == 's' || ch == 'z') z->k++;
   }
   } else if (m(z) == 1 && cvc(z, z->k))
   setto(z,
   "\01"
   "e");
   }
   }


    inline  void step1c(   __global struct stemmer *z) {
   if (ends(z,
   "\01"
   "y") &&
   vowelinstem(z))
   z->b[z->k] = 'i';
   }

 inline     void step2(   __global struct stemmer *z) {
   switch (z->b[z->k - 1]) {
case 'a':
if (ends(z,
			"\07"
			"ational")) {
	r(z,
			"\03"
			"ate");
	break;
}
if (ends(z,
			"\06"
			"tional")) {
	r(z,
			"\04"
			"tion");
	break;
}
break;
case 'c':
if (ends(z,
			"\04"
			"enci")) {
	r(z,
			"\04"
			"ence");
	break;
}
if (ends(z,
			"\04"
			"anci")) {
	r(z,
			"\04"
			"ance");
	break;
}
break;
case 'e':
if (ends(z,
			"\04"
			"izer")) {
	r(z,
			"\03"
			"ize");
	break;
}
break;
case 'l':
if (ends(z,
			"\03"
			"bli")) {
	r(z,
			"\03"
			"ble");
	break;
}


if (ends(z,
			"\04"
			"alli")) {
	r(z,
			"\02"
			"al");
	break;
}
if (ends(z,
			"\05"
			"entli")) {
	r(z,
			"\03"
			"ent");
	break;
}
if (ends(z,
			"\03"
			"eli")) {
	r(z,
			"\01"
			"e");
	break;
}
if (ends(z,
			"\05"
			"ousli")) {
	r(z,
			"\03"
			"ous");
	break;
}
break;
case 'o':
if (ends(z,
			"\07"
			"ization")) {
	r(z,
			"\03"
			"ize");
	break;
}
if (ends(z,
			"\05"
			"ation")) {
	r(z,
			"\03"
			"ate");
	break;
}
if (ends(z,
			"\04"
			"ator")) {
	r(z,
			"\03"
			"ate");
	break;
}
break;
case 's':
if (ends(z,
			"\05"
			"alism")) {
	r(z,
			"\02"
			"al");
	break;
}
if (ends(z,
			"\07"
			"iveness")) {
	r(z,
			"\03"
			"ive");
	break;
}
if (ends(z,
			"\07"
			"fulness")) {
	r(z,
			"\03"
			"ful");
	break;
}
if (ends(z,
			"\07"
			"ousness")) {
	r(z,
			"\03"
			"ous");
	break;
}
break;
case 't':
if (ends(z,
			"\05"
			"aliti")) {
	r(z,
			"\02"
			"al");
	break;
}
if (ends(z,
			"\05"
			"iviti")) {
	r(z,
			"\03"
			"ive");
	break;
}
if (ends(z,
			"\06"
			"biliti")) {
	r(z,
			"\03"
			"ble");
	break;
}
break;
case 'g':
if (ends(z,
			"\04"
			"logi")) {
	r(z,
			"\03"
			"log");
	break;
}

}
}

 inline  void step3(   __global struct stemmer *z) {
	switch (z->b[z->k]) {
		case 'e':
			if (ends(z,
						"\05"
						"icate")) {
				r(z,
						"\02"
						"ic");
				break;
			}
			if (ends(z,
						"\05"
						"ative")) {
				r(z,
						"\00"
						"");
				break;
			}
			if (ends(z,
						"\05"
						"alize")) {
				r(z,
						"\02"
						"al");
				break;
			}
			break;
		case 'i':
			if (ends(z,
						"\05"
						"iciti")) {
				r(z,
						"\02"
						"ic");
				break;
			}
			break;
		case 'l':
			if (ends(z,
						"\04"
						"ical")) {
				r(z,
						"\02"
						"ic");
				break;
			}
			if (ends(z,
						"\03"
						"ful")) {
				r(z,
						"\00"
						"");
				break;
			}
			break;
		case 's':
			if (ends(z,
						"\04"
						"ness")) {
				r(z,
						"\00"
						"");
				break;
			}
			break;
	}
}


 inline  void step4(   __global struct stemmer *z) {
	switch (z->b[z->k - 1]) {
		case 'a':
			if (ends(z,
						"\02"
						"al"))
				break;
			return;
		case 'c':
			if (ends(z,
						"\04"
						"ance"))
				break;
			if (ends(z,
						"\04"
						"ence"))
				break;
			return;
		case 'e':
			if (ends(z,
						"\02"
						"er"))
				break;
			return;
		case 'i':
			if (ends(z,
						"\02"
						"ic"))
				break;
			return;
		case 'l':
			if (ends(z,
						"\04"
						"able"))
				break;
			if (ends(z,
						"\04"
						"ible"))
				break;
			return;
		case 'n':
			if (ends(z,
						"\03"
						"ant"))
				break;
			if (ends(z,
						"\05"
						"ement"))
				break;
			if (ends(z,
						"\04"
						"ment"))
				break;
			if (ends(z,
						"\03"
						"ent"))
				break;
			return;
		case 'o':
			if (ends(z,
						"\03"
						"ion") &&
					(z->b[z->j] == 's' || z->b[z->j] == 't'))
				break;
			if (ends(z,
						"\02"
						"ou"))
				break;
			return;
		case 's':
			if (ends(z,
						"\03"
						"ism"))
				break;
			return;
		case 't':
			if (ends(z,
						"\03"
						"ate"))
				break;
			if (ends(z,
						"\03"
						"iti"))
				break;
			return;
		case 'u':
			if (ends(z,
						"\03"
						"ous"))
				break;
			return;
		case 'v':
			if (ends(z,
						"\03"
						"ive"))
				break;
			return;
		case 'z':
			if (ends(z,
						"\03"
						"ize"))
				break;
			return;
		default:
			return;
	}
	if (m(z) > 1) z->k = z->j;
}


inline void step5(__global struct stemmer *z) {
//	char *b = z->b;
	z->j = z->k;
	if (z->b[z->k] == 'e') {
		int a = m(z);
		if (a > 1 || (a == 1 && !cvc(z, z->k - 1))) 
		z->k--;
	}
	if (z->b[z->k] == 'l' && doublec(z, z->k) && m(z) > 1) z->k--;
}

__kernel void stem_gpu(__global void *stem_list, int words) {
	int tid = get_group_id(0) * get_local_size(0) + get_local_id(0);
	if (tid < words) {
		if (((__global struct stemmer *)stem_list)[tid].k <= 1) {
			return;
		}
		
		   step1ab(&(((__global struct stemmer *)stem_list)[tid]));
		   step1c(&(((__global struct stemmer *)stem_list)[tid]));
		   step2(&(((__global struct stemmer *)stem_list)[tid]));
		   step3(&(((__global struct stemmer *)stem_list)[tid]));
			
		   step4(&(((__global struct stemmer *)stem_list)[tid]));
		   step5(&(((__global struct stemmer *)stem_list)[tid]));
		   ((__global struct stemmer *)stem_list)[tid].b[((__global struct stemmer *)stem_list)[tid].k + 1] = 0;
	}
}
