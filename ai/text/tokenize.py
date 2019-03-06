import regex as re

FLAGS = re.MULTILINE | re.DOTALL

EYES_PATTERN = r"[8:=;]"
NOSE_PATTERN = r"['`\-]?"


class TwitterGloveTokenizer:
    # function so code less repetitive
    def _re_sub(self, text, pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)

    def tokenize(self, text):
        text = self._re_sub(text, r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
        text = self._re_sub(text, r"@\w+", "<user>")
        text = self._re_sub(text, r"{}{}[)dD]+|[)dD]+{}{}".format(EYES_PATTERN, NOSE_PATTERN, NOSE_PATTERN, EYES_PATTERN), "<smile>")
        text = self._re_sub(text, r"{}{}p+".format(EYES_PATTERN, NOSE_PATTERN), "<lolface>")
        text = self._re_sub(text, r"{}{}\(+|\)+{}{}".format(EYES_PATTERN, NOSE_PATTERN, NOSE_PATTERN, EYES_PATTERN), "<sadface>")
        text = self._re_sub(text, r"{}{}[\/|l*]".format(EYES_PATTERN, NOSE_PATTERN), "<neutralface>")
        text = self._re_sub(text, r"/", " / ")
        text = self._re_sub(text, r"<3", "<heart>")
        text = self._re_sub(text, r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
        text = self._re_sub(text, r"#\S+", self._hashtag)
        text = self._re_sub(text, r"([!?.]){2,}", r"\1 <repeat>")
        text = self._re_sub(text, r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")
        text = self._re_sub(text, r"([A-Z]){2,}", self._allcaps)
        text = text.lower()
        text = text.split()
        return text

    def _hashtag(self, text):
        text = text.group()
        hashtag_body = text[1:]
        if hashtag_body.isupper():
            result = "<hashtag> {} <allcaps>".format(hashtag_body)
        else:
            result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
        return result

    def _allcaps(self, text):
        text = text.group()
        return text.lower() + " <allcaps>"
