# Start

This is a galaxy dynamical modeling code. You can find its doc here: `https://www.mpe.mpg.de/resources/IR/DYSMALPY/index.html`. However, it is relatively slow. I want you to make it faster with two potential improvements,
- Use JAX to speedup the model calculation with GPU. This matching has NVIDIA 4090. Work in `conda activate dysmalpy-jax` and JAX is installed.
- Replace the model and parameters which directly use the astropy classes with our customized class so that they are lighter and faster.

Please evaluate the strategy carefully. I tried to ask claude code to learn dysmalpy and write the code from the scratch in `/home/shangguan/Softwares/my_modules/dysmalpy-jax`. You can check the code there for the customized parameter class and the JAX implementation. However, I realize that it is still better to revise the original code. Let's keep record of this development for JAX in the git `dev_jax` branch.

Think carefully and make a plan.md in `/home/shangguan/Softwares/my_modules/dysmalpy/dev`, which is your working folder for the development.


# Unit tests

After implementing all the changes, write down what you have down in a develop_note.md in `/home/shangguan/Softwares/my_modules/dysmalpy/dev`. Then, update the unit tests (`/home/shangguan/Softwares/my_modules/dysmalpy/tests`) so that we can fully test the updated functions. Think carefully and make a plan. Update the develop_note.md with all the todo list. Make a plan first.


# Complete the plan

Check the develop_note.md and see what is left to finish.


# Compare with the original dysmalpy

I suggest you to fix the JAX/Numpy problem. Now, let's fully use JAX as much as possible. Then, make a script that derive the model results, that is used finally to calculate the chisquared with the real data, and compare the new and originaly dysmalpy. Make sure that the model results are identical at this level. Check carefully the code and make a plan.


# Test tutorial cases

Investigate the different tutorials in this page carefully, `https://www.mpe.mpg.de/resources/IR/DYSMALPY/tutorials/fitting/index.html` Make a two-stage plan. I want you to make a plan to implement the tests to run the examples and compare with the tutorial results. I want you to first test MPFIT and ADAM optimizer with the 1D fitting to make sure that the fitting is operational. Then, we proceed to test the fitting with all the modes.


# Speedup evaluation

Now, as the MPFIT can provide us correct results for all the 1-3 D data. I want you to do a test of the speedup with the MPFIT. I want you to update the speedup report (/home/shangguan/Softwares/my_modules/dysmalpy/dev/speedup.md) with real data that you used for the above tests. Give me a plan.

Make a plan to check the most important speed bottlenecks. Think carefully and make a plan and update the /home/shangguan/Softwares/my_modules/dysmalpy/dev/speedup.md


# Demo

Please follow `https://www.mpe.mpg.de/resources/IR/DYSMALPY/tutorials/fitting/dysmalpy_example_fitting_wrapper_2D.html` and make a fitting demo using the 2D fitting mode with the new dysmalpy. I want a python script that can run and produce the figures and a markdown instruction that basically follow the website example including the figures. Run the script and make the markdown file, so that I can check the markdown file to learn how to use the new code. Give me a plan first.

Now, it seems that the MCMC and dynesty fitting code works. Let's run the demo with the MCMC fitter again in `/home/shangguan/Softwares/my_modules/dysmalpy/demo`.


# Parallization

If there is not a clean solution, let's implement the option D (only allow ncpu=1 for dynesty and emcee). On the otherhand, can we use https://jaxns.readthedocs.io/en/latest/ for nested sampling which presumably can be faster by fully supporting JAX? Check the doc of JAXNS carefully and give me a plan.

The MCMC fitting results are incorrect. You can see your demo report. The acceptance rate is 0 and the result plots are empty (I mean no meaningful results except for the colorbar). Understand what when wrong. Also, in parallel, please make a demo using jaxns to fit the same data. Give me a plan.


# GPU compatible

Now, check if you can run the demos with GPU instead of the CPU. Start with the MPFIT, MCMC, and JAXNS.

Now, it seems that MCMC does not work well with multiple cores. Please check the code carefully and fix the problem. Work on the problem in `/home/shangguan/Softwares/my_modules/dysmalpy/dev`. Read the `/home/shangguan/Softwares/my_modules/dysmalpy/dev/develop_note.md` and `/home/shangguan/Softwares/my_modules/dysmalpy/dev/development_note.md` first. Write down your plan. Actually, merge everything in development_note.md and remove develop_note.md. Give me a plan first.