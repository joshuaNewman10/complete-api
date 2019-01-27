import regex as re

FLAGS = re.MULTILINE | re.DOTALL

eyes = r"[8:=;]"
nose = r"['`\-]?"


def hashtag(text):
    text = text.group()
    hashtag_body = text[1:]
    if hashtag_body.isupper():
        result = "<hashtag> {} <allcaps>".format(hashtag_body)
    else:
        result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
    return result


def allcaps(text):
    text = text.group()
    return text.lower() + " <allcaps>"


class TwitterGloveTokenizer:
    # function so code less repetitive
    def _re_sub(self, text, pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    def tokenize(self, text):
        text = self._re_sub(text, r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
        text = self._re_sub(text, r"@\w+", "<user>")
        text = self._re_sub(text, r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
        text = self._re_sub(text, r"{}{}p+".format(eyes, nose), "<lolface>")
        text = self._re_sub(text, r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
        text = self._re_sub(text, r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
        text = self._re_sub(text, r"/", " / ")
        text = self._re_sub(text, r"<3", "<heart>")
        text = self._re_sub(text, r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
        text = self._re_sub(text, r"#\S+", hashtag)
        text = self._re_sub(text, r"([!?.]){2,}", r"\1 <repeat>")
        text = self._re_sub(text, r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")
        text = self._re_sub(text, r"([A-Z]){2,}", allcaps)
        text = text.lower()
        text = text.split()
        return text
