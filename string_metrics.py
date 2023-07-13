import jellyfish as jf
import Levenshtein as lev


def damerau_levenshtein(a, b):
    return jf.damerau_levenshtein_distance(a, b)


def match_dam_lev(a, b):
    return 1 - jf.damerau_levenshtein_distance(a, b) / lmax(a, b)


def levenshtein(a, b):
    return lev.distance(a, b)


def match_lev(a, b):
    return 1 - lev.distance(a, b) / lmax(a, b) if len(a) != 0 else 0


def inter(a, b):
    return len(set(a).intersection(set(b)))


from difflib import SequenceMatcher


def lcs(a, b):
    s = SequenceMatcher(None, a, b)
    lcs_ = "".join([a[block.a:(block.a + block.size)] for block in s.get_matching_blocks()])
    return len(lcs_)


def lcs_match(a, b):
    lmin_a_b = lmin(a, b)
    if lmin_a_b:
        return max(lcs(a, b), lcs(b, a)) / lmin_a_b
    else:
        return 0




def hamming(a, b):
    m = min(len(a), len(b))
    return sum(a[i] == b[i] for i in range(m))


def hamming_match(a, b):
    lmin_a_b = lmin(a, b)
    if lmin_a_b:
        return hamming(a, b) / lmin_a_b
    else:
        return 0


def jaro(a, b):
    return jf.jaro_similarity(a, b)


def jaro_len_filter(a, b):
    return (len(a) / len(b) + 2) / 3


def jaro_char_filter(a, b):
    m = min(len(a), len(b)) - min_diff(a, b)
    return (m / len(a) + m / len(b) + 1) / 3


def lens(a, b):
    return len(a) + len(b)


def area(a, b):
    return len(set(a).union(set(b)))


def lsym_diff(a, b):
    return len(set(a).symmetric_difference(set(b)))


def min_diff(a, b):
    set_a, set_b = set(a), set(b)
    diff_a = len(set_a.difference(set_b))
    diff_b = len(set_b.difference(set_a))
    return min(diff_b, diff_a)


def dsc(a, b):
    return 2 * inter(a, b) / lens(a, b)


def jaccard_index(a, b):
    return inter(a, b) / area(a, b) if area(a, b) else 0


def token_jaccard_index(a, b):
    A, B = a.split(), b.split()
    return jaccard_index(A, B)


from token_metrics import soft_inter as tokens_soft_inter
def soft_jaccard_index(a, b, sim_fun=jaro, theta=0.85): #<- change name"
    A, B = a.split(), b.split()
    soft_inter = tokens_soft_inter(A, B, sim_fun, theta)
    return len(soft_inter)/area(A, B) if area(A, B) else 0


def lmin(a, b):
    return min(len(a), len(b))

def lmax(a, b):
    return max(len(a), len(b))


def overlap(a, b):
    return inter(a, b) / lmin(a, b)


def gram2(a, b):
    ag = [a[i:i + 2] for i in range(len(a) - 1)]
    bg = [b[i:i + 2] for i in range(len(b) - 1)]
    return sum((g in ag) and (g in bg) for g in ag + bg)


def ldiff(a, b):
    return abs(len(a) - len(b))


from collections import Counter
def sym_diff_count_list(a, b):
    listA = list(a)
    listB = list(b)
    diff_list_1 = list((Counter(listA) - Counter(listB)).elements())
    diff_list_2 = list((Counter(listB) - Counter(listA)).elements())
    return sorted(diff_list_1 + diff_list_2)


def sym_diff(a, b):
    return sorted(list(set(a).symmetric_difference(set(b))))


def sym_diff_count_set(a, b):
    listA = list(a)
    listB = list(b)
    diff_list_1 = list((Counter(listA) - Counter(listB)).elements())
    diff_list_2 = list((Counter(listB) - Counter(listA)).elements())
    return sorted(list(set(diff_list_1 + diff_list_2)))


def lens_match(a, b):
    N = len(a) + len(b)
    if N > 0:
        return 1 - abs(len(a)-len(b)) / N
    else:
        return 0


from nltk.stem import SnowballStemmer, RegexpStemmer #WordNetLemmatizer
_stem = SnowballStemmer('english')
_regexp = RegexpStemmer('ition$|ae$|ism$|fe$|stic$|ature$|ve$|ly$|ues$|ography$|ive$|ative$|ian$|cation$|s$|sis$|te$|ling$|ory$|ions$|ite$|ic$|ification$|obicity$|al$|eity$|led$|ers$|ing$|le$|cs$|ally$|cal$|ration$|ilicity$|ion$|tic$|osis$|ry$|ability$|ors$|cy$|an$|etical$|inal$|ifier$|zation$|ality$|y$|ices$|or$|ging$|tics$|tible$|ible$|ary$|logy$|ice$|ping$|ex$|ivity$|onization$|icity$|ization$|odoly$|ning$|uous$|ity$|ized$|ty$|e$|x$|ouse$|ics$|isation$|t$|n$|vity$|ier$|se$|bility$|ned$|ial$|ine$|ene$|es$|um$|des$|ng$|ri$|ming$|ren$|ium$|arity$|is$|ler$|ar$|ous$|ility$|ness$|ment$|ionism$|a$|m$|ple$|ne$|r$|i$|able$|lic$|d$|ting$|ery$|ology$|izing$|ces$|izens$|eries$|phy$|ants$|en$|graphy$|ed$|ical$|iness$|er$|em$|us$|istic$|tion$|ies$|on$|ation$|ce$', min=5)

#from pattern.text.en import singularize
from string import punctuation

punctuation += "-––-–"

from nltk.corpus import stopwords

english_stopwords = stopwords.words("english")


def replace_multiple(string, list_replace, replace_ch):
    for ch in list_replace:
        if ch in string:
            string = string.replace(ch, replace_ch)
    return string


def replace_by_dict(string, dict_replace):
    for k, v in dict_replace.items():
        string = string.replace(k, v)
    return string


def uv_ratio(a, b):
    len_A = len(a)
    len_B = len(b)
    if 0 < len_A <= len_B:
        return len_A / len_B
    else:
        return - len_B / len_A if len_A > 0 else 0


def uv_ratio_tokens(a, b):
    A = a.split()
    B = b.split()
    return uv_ratio(A, B)


def uv_ratio_injection(a, b):
    inter_A_B = set(a).intersection(set(b))
    return uv_ratio(inter_A_B.intersection(set(a)), a)


def uv_ratio_injection_tokens(a, b):
    A, B = a.split(), b.split()
    return uv_ratio_injection(A, B)


def soft_uv_ratio_injection(a, b, sim_fun=jaro, theta=0.85):
    inter_A_B = tokens_soft_inter(a, b, sim_fun, theta)
    return uv_ratio(inter_A_B.intersection(set(a)), a)


def soft_uv_ratio_injection_tokens(a, b, sim_fun=jaro, theta=0.85):
    A, B = a.split(), b.split()
    return soft_uv_ratio_injection(A, B, sim_fun, theta)

from unidecode import unidecode
def get_trans(s, by="all"):
    s = " ".join(s.lower().split())

    if by == "wpunct_sort_stem":
        s_punct = " ".join(replace_multiple(s, punctuation, " ").split())
        s_punct_sort = " ".join(sorted(s_punct.split()))
        return " ".join([_stem.stem(w) for w in s_punct_sort.split()])

    if by == "diacritic_punct_stopw_regex_sort":
        s_diacritic = unidecode(s)
        s_diacritic_punct = " ".join(replace_multiple(s_diacritic, punctuation, " ").split())
        s_diacritic_punct_stopw = " ".join(filter(lambda item: item not in english_stopwords, s_diacritic_punct.split()))
        s_diacritic_punct_stopw_regex = " ".join([_regexp.stem(w) for w in s_diacritic_punct_stopw.split()])
        s_diacritic_punct_stopw_regex_sort = " ".join(sorted(s_diacritic_punct_stopw_regex.split()))
        return s_diacritic_punct_stopw_regex_sort

    elif by == "simply":
        s_diacritic = unidecode(s)
        s_diacritic_punct = " ".join(replace_multiple(s_diacritic, punctuation, " ").split())
        s_diacritic_punct_stopw = " ".join(filter(lambda item: item not in english_stopwords, s_diacritic_punct.split()))
        s_diacritic_punct_stopw_regex = " ".join([_regexp.stem(w) for w in s_diacritic_punct_stopw.split()])
        s_diacritic_punct_stopw_regex_sort = " ".join(sorted(s_diacritic_punct_stopw_regex.split()))
        return [s_diacritic, s_diacritic_punct, s_diacritic_punct_stopw, s_diacritic_punct_stopw_regex,
                s_diacritic_punct_stopw_regex_sort]
    else:
        s_punct = " ".join(replace_multiple(s, punctuation, " ").split())

        s_stopw = " ".join(filter(lambda item: item not in english_stopwords, s.split()))

        s_sort = " ".join(sorted(s.split()))

        s_singular = " ".join([singularize(w) for w in s.split()])

        s_stem = " ".join([_stem.stem(w) for w in s.split()])

        s_regexp = " ".join([_regexp.stem(w) for w in s.split()])

        s_abb = "".join([w[0] for w in s.split()])

        s_diacritic = unidecode(s)

        s_punct_singular = " ".join([singularize(w) for w in s_punct.split()])

        s_punct_stem = " ".join([_stem.stem(w) for w in s_punct.split()])

        s_punct_regexp = " ".join([_regexp.stem(w) for w in s_punct.split()])

        s_punct_abb = "".join([w[0] for w in s_punct.split()])

        s_punct_stopw = " ".join(filter(lambda item: item not in english_stopwords, s_punct.split()))

        s_punct_sort = " ".join(sorted(s_punct.split()))

        s_punct_diacritic = unidecode(s_punct)

        s_stopw_singular = " ".join([singularize(w) for w in s_stopw.split()])

        s_stopw_stem = " ".join([_stem.stem(w) for w in s_stopw.split()])

        s_stopw_regexp = " ".join([_regexp.stem(w) for w in s_stopw.split()])

        s_stopw_abb = "".join([w[0] for w in s_stopw.split()])

        s_stopw_sort = " ".join(sorted(s_stopw.split()))

        s_stopw_diacritic = unidecode(s_stopw)

        s_sort_singular = " ".join([singularize(w) for w in s_sort.split()])

        s_sort_stem = " ".join([_stem.stem(w) for w in s_sort.split()])

        s_sort_regexp = " ".join([_regexp.stem(w) for w in s_sort.split()])

        s_sort_abb = "".join([w[0] for w in s_sort.split()])

        s_sort_diacritic = unidecode(s_sort)

        s_punct_sort_singular = " ".join([singularize(w) for w in s_punct_sort.split()])

        s_punct_sort_stem = " ".join([_stem.stem(w) for w in s_punct_sort.split()])

        s_punct_sort_regexp = " ".join([_regexp.stem(w) for w in s_punct_sort.split()])

        s_punct_sort_abb = "".join([w[0] for w in s_punct_sort.split()])

        s_punct_sort_diacritic = unidecode(s_punct_sort)

        s_stopw_sort_singular = " ".join([singularize(w) for w in s_stopw_sort.split()])

        s_stopw_sort_stem = " ".join([_stem.stem(w) for w in s_stopw_sort.split()])

        s_stopw_sort_regexp = " ".join([_regexp.stem(w) for w in s_stopw_sort.split()])

        s_stopw_sort_abb = "".join([w[0] for w in s_stopw_sort.split()])

        s_stopw_sort_diacritic = unidecode(s_stopw_sort)

        s_punct_stopw_singular = " ".join([singularize(w) for w in s_punct_stopw.split()])

        s_punct_stopw_stem = " ".join([_stem.stem(w) for w in s_punct_stopw.split()])

        s_punct_stopw_regexp = " ".join([_regexp.stem(w) for w in s_punct_stopw.split()])

        s_punct_stopw_abb = "".join([w[0] for w in s_punct_stopw.split()])

        s_punct_stopw_sort = " ".join(sorted(s_punct_stopw.split()))

        s_punct_stopw_diacritic = unidecode(s_punct_stopw)

        s_punct_stopw_sort_singular = " ".join([singularize(w) for w in s_punct_stopw_sort.split()])

        s_punct_stopw_sort_stem = " ".join([_stem.stem(w) for w in s_punct_stopw_sort.split()])

        s_punct_stopw_sort_regexp = " ".join([_regexp.stem(w) for w in s_punct_stopw_sort.split()])

        s_punct_stopw_sort_abb = "".join([w[0] for w in s_punct_stopw_sort.split()])

        s_punct_stopw_sort_diacritic = unidecode(s_punct_stopw_sort)

        return [s,
                s_punct, s_stopw, s_sort, s_singular, s_stem, s_regexp, s_abb, s_diacritic,
                s_punct_singular, s_punct_stem, s_punct_regexp, s_punct_abb, s_punct_stopw, s_punct_sort,
                s_punct_diacritic,
                s_stopw_singular, s_stopw_stem, s_stopw_regexp, s_stopw_abb, s_stopw_sort, s_stopw_diacritic,
                s_sort_singular, s_sort_stem, s_sort_regexp, s_sort_abb, s_sort_diacritic,
                s_punct_stopw_singular, s_punct_stopw_stem, s_punct_stopw_regexp, s_punct_stopw_abb, s_punct_stopw_sort, s_punct_stopw_diacritic,
                s_punct_sort_singular, s_punct_sort_stem, s_punct_sort_regexp, s_punct_sort_abb, s_punct_sort_diacritic,
                s_stopw_sort_singular, s_stopw_sort_stem, s_stopw_sort_regexp, s_stopw_sort_abb, s_stopw_sort_diacritic,
                s_punct_stopw_sort_singular, s_punct_stopw_sort_stem, s_punct_stopw_sort_regexp, s_punct_stopw_sort_abb, s_punct_stopw_sort_diacritic
                ]

def sim_trans_str_measure(s, t, by="wpunct_sort_stem"):
    return match_lev(get_trans(s, by=by), get_trans(t, by=by))