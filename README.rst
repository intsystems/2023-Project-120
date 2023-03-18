|test| |codecov| |docs|

.. |test| image:: https://github.com/intsystems/ProjectTemplate/workflows/test/badge.svg
    :target: https://github.com/intsystems/ProjectTemplate/tree/master
    :alt: Test status
    
.. |codecov| image:: https://img.shields.io/codecov/c/github/intsystems/ProjectTemplate/master
    :target: https://app.codecov.io/gh/intsystems/ProjectTemplate
    :alt: Test coverage
    
.. |docs| image:: https://github.com/intsystems/ProjectTemplate/workflows/docs/badge.svg
    :target: https://intsystems.github.io/ProjectTemplate/
    :alt: Docs status


.. class:: center

    :Название исследуемой задачи: Differentiable algorithm for searching ensembles of deep learning models with diversity control
    :Тип научной работы: M1P
    :Автор: Peter Babkin Konstantinovich
    :Научный руководитель: степень, Фамилия Имя Отчество
    :Научный консультант(при наличии): степень, Фамилия Имя Отчество

Abstract
========

This paper is developed to introduce a new method of creating ensembles of deep learning models. 
Many modern researches were focused on creating effective and efficient algorithms of differentiable architecture search,
missing oppotunity to create ensembles of less sophisticated deep learning models. This approach gives impressive results
as it was shown in few modern papers. In our research we investigate an algorithm of sampling deep learning models using
hypernetwork, which controls diversity of the models. Distinction between two models is measured in terms of Jensen–Shannon
divergence which keeps the algorythm differentiable. To evaluate the performance of the proposed algorithm, we conducted
experiments on the Fashion-MNIST and CIFAR-10 datasets and compare the resulting ensembles with ones sampled by other
searching algorithms.

Research publications
===============================
1. 

Presentations at conferences on the topic of research
================================================
1. 

Supplementations
================
1. `Link review <https://docs.google.com/document/d/1-P76pFjZ2E4BIjLVU8KY1NC7g1Qt-YFh6zX-V67FTUU/edit>`_.
2. `Overleaf project <https://www.overleaf.com/3228135464pjqvcbkvrgwb>`_.
3. `Overleaf project <https://www.overleaf.com/project/640f5ce53337a9708acf21c3>_.

Software modules developed as part of the study
======================================================
1. A python package *mylib* with all implementation `here <https://github.com/intsystems/ProjectTemplate/tree/master/src>`_.
2. A code with all experiment visualisation `here <https://github.comintsystems/ProjectTemplate/blob/master/code/main.ipynb>`_. Can use `colab <http://colab.research.google.com/github/intsystems/ProjectTemplate/blob/master/code/main.ipynb>`_.
