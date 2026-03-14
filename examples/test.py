import pandas as pd
from pyrstat import lm, ivreg, summary_ivreg

data = pd.read_csv("/demand.csv", sep=",")

reg_iv = ivreg("log(q.butter) ~ log(p.butter) + log(income) + log(p.marg) | log(income) + log(c.butter) + log(c.marg)", data)

summary_ivreg(reg_iv)
