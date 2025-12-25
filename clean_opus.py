from langdetect import detect

seen = set()

def clean(tsv):
    cleaned_pairs = []
    s_candidate = None
    t_candidate = None
    with open(tsv) as f:
        for line in f:
            line = line.strip()
            if line.startswith("(src)=") and len(line) > 6:
                s_candidate = line[line.find('>') + 1:].strip()
            elif line.startswith("(trg)=") and len(line) > 6:
                t_candidate = line[line.find('>') + 1:].strip()

            if s_candidate and t_candidate:
                s = s_candidate
                t = t_candidate
                s_candidate = None
                t_candidate = None

                if not s or not t:
                    continue
                if len(s.split()) > 200 or len(t.split()) > 200:
                    continue
                if max(len(s.split()), len(t.split())) / max(1, min(len(s.split()), len(t.split()))) > 3:
                    continue
                try:
                    if detect(s) != "en":
                        continue
                except:
                    continue
                
                cleaned_pairs.append((s, t))
    return cleaned_pairs
