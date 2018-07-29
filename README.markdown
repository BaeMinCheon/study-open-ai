
---

# Setup

---

1. download `Anaconda for Windows` in https://www.anaconda.com/download/  

no matter whether it is `Python 3.X` or `Python 2.7` but I recommend you `Python 2.7` side

2. install `Anaconda` with default properties  

no need to adjust from default check box

3. execute `Anaconda Prompt`
4. type `conda create -n openai pip python=3.5`
5. type `activate openai`
6. type `pip install gym`
7. _PROFIT !_

---

# Additional
_when you need to ues OpenAI with Tensorflow_

---

1. execute `Anaconda Prompt`
2. type `activate openai`
3. type `pip install --ignore-installed --upgrade tensorflow-gpu`  

because I gonna use tensorflow supporting gpu

4. _PROFIT !_