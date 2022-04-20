# FM_Scoring_Model
Practice of score prediction based on FM model<br />
<br />
Contains data processing and data analysis modules<br />
<br />
Including:<br />
<br />
1 Average online time and Popularity Regression analysis<br />
2 Slicing, and date data processing<br />
3 Linear or binary problem<br />

<br />
## Another work<br />
This work focus on **[matrix decomposition]**, we all know that MF could be viewed as a special case of FM, another way to say it, FM model could be used in **[matrix decomposition]**.<br />
But if we want to put constraints on the result of matrix factorization, FM, or embedding ways may not work effectively. So we can transforme this problem into constrained programming solution problems.<br />
Here is a simple attempt to use the Generalized reduced gradient method.<br />
<br />
The constraint we set here is that the residual is 0, and the result of matrix splitting must be a coincidence sum.<br />
The details are in the file named SRG.<br />
