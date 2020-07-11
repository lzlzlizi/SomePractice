Minimize
 1 q0 + 1 q1 + 1 q2 + 1 q3 + 1 q4 + 1 q5 + 1 q6 + 1 q7 + 1 q8 + 1 q9 + 1 q10 + 1 q11 + 1 q12 + 1 q13 + 1 q14 + 1 q15 + 1 q16 + 1 q17 + 1 q18 + 1 q19 + 1 q20 + 1 q21 + 1 q22 + 1 q23 + 1 q24 + 1 q25 + 1 q26 + 1 q27 + 1 q28 + 1 q29 + 1 q30 + 1 q31 + 1 q32 + 1 q33 + 1 q34 + 1 q35 + 1 q36 + 1 q37 + 1 q38 + 1 q39 + 1 q40 + 1 q41 + 1 q42 + 1 q43 + 1 q44 + 1 q45 + 1 q46 + 1 q47 + 1 q48 + 1 q49 + 1 q50 + 1 q51 + 1 q52 + 1 q53 + 1 q54 + 1 q55 + 1 q56 + 1 q57 + 1 q58 + 1 q59 + [ 3 l0 ^ 2 + 3 l1 ^ 2 + 3 l2 ^ 2 + 3 l3 ^ 2 + 3 l4 ^ 2 + 3 l5 ^ 2 + 3 l6 ^ 2 + 3 l7 ^ 2 + 3 l8 ^ 2 + 3 l9 ^ 2 + 3 l10 ^ 2 + 3 l11 ^ 2 + 3 l12 ^ 2 + 3 l13 ^ 2 + 3 l14 ^ 2 + 3 l15 ^ 2 + 3 l16 ^ 2 + 3 l17 ^ 2 + 3 l18 ^ 2 + 3 l19 ^ 2 + 3 l20 ^ 2 + 3 l21 ^ 2 + 3 l22 ^ 2 + 3 l23 ^ 2 + 3 l24 ^ 2 + 3 l25 ^ 2 + 3 l26 ^ 2 + 3 l27 ^ 2 + 3 l28 ^ 2 + 3 l29 ^ 2 + 3 l30 ^ 2 + 3 l31 ^ 2 + 3 l32 ^ 2 + 3 l33 ^ 2 + 3 l34 ^ 2 + 3 l35 ^ 2 + 3 l36 ^ 2 + 3 l37 ^ 2 + 3 l38 ^ 2 + 3 l39 ^ 2 + 3 l40 ^ 2 + 3 l41 ^ 2 + 3 l42 ^ 2 + 3 l43 ^ 2 + 3 l44 ^ 2 + 3 l45 ^ 2 + 3 l46 ^ 2 + 3 l47 ^ 2 + 3 l48 ^ 2 + 3 l49 ^ 2 + 3 l50 ^ 2 + 3 l51 ^ 2 + 3 l52 ^ 2 + 3 l53 ^ 2 + 3 l54 ^ 2 + 3 l55 ^ 2 + 3 l56 ^ 2 + 3 l57 ^ 2 + 3 l58 ^ 2 + 3 l59 ^ 2 + 3 l60 ^ 2 + 3 l61 ^ 2 + 3 l62 ^ 2 ] / 2
Subject To
 vx0 = 10000
 vy0 = 10000
 pa0 - 9 a0 <= 0
 pa0 - l0 <= 0
 pa0 - l0 - 9 a0 >= -9
 pb0 - 9 b0 <= 0
 pb0 - l0 <= 0
 pb0 - l0 - 9 b0 >= -9
 pc0 - 9 c0 <= 0
 pc0 - l0 <= 0
 pc0 - l0 - 9 c0 >= -9
 pd0 - 9 d0 <= 0
 pd0 - l0 <= 0
 pd0 - l0 - 9 d0 >= -9
 vx42 - vx40 - pa0 + pb0 = 0
 vy42 - vy40 - pc0 + pd0 = 0
 a0 + b0 <= 1
 c0 + d0 <= 1
 a0 = 0
 b0 = 1
 c0 = 0
 pa1 - 9 a1 <= 0
 pa1 - l1 <= 0
 pa1 - l1 - 9 a1 >= -9
 pb1 - 9 b1 <= 0
 pb1 - l1 <= 0
 pb1 - l1 - 9 b1 >= -9
 pc1 - 9 c1 <= 0
 pc1 - l1 <= 0
 pc1 - l1 - 9 c1 >= -9
 pd1 - 9 d1 <= 0
 pd1 - l1 <= 0
 pd1 - l1 - 9 d1 >= -9
 vx5 - vx42 - pa1 + pb1 = 0
 vy5 - vy42 - pc1 + pd1 = 0
 a1 + b1 <= 1
 c1 + d1 <= 1
 a1 = 0
 b1 = 1
 c1 = 0
 pa2 - 9 a2 <= 0
 pa2 - l2 <= 0
 pa2 - l2 - 9 a2 >= -9
 pb2 - 9 b2 <= 0
 pb2 - l2 <= 0
 pb2 - l2 - 9 b2 >= -9
 pc2 - 9 c2 <= 0
 pc2 - l2 <= 0
 pc2 - l2 - 9 c2 >= -9
 pd2 - 9 d2 <= 0
 pd2 - l2 <= 0
 pd2 - l2 - 9 d2 >= -9
 vx63 - vx5 - pa2 + pb2 = 0
 vy63 - vy5 - pc2 + pd2 = 0
 a2 + b2 <= 1
 c2 + d2 <= 1
 a2 = 0
 d2 = 0
 b2 = 1
 pa3 - 9 a3 <= 0
 pa3 - l3 <= 0
 pa3 - l3 - 9 a3 >= -9
 pb3 - 9 b3 <= 0
 pb3 - l3 <= 0
 pb3 - l3 - 9 b3 >= -9
 pc3 - 9 c3 <= 0
 pc3 - l3 <= 0
 pc3 - l3 - 9 c3 >= -9
 pd3 - 9 d3 <= 0
 pd3 - l3 <= 0
 pd3 - l3 - 9 d3 >= -9
 vx15 - vx63 - pa3 + pb3 = 0
 vy15 - vy63 - pc3 + pd3 = 0
 a3 + b3 <= 1
 c3 + d3 <= 1
 c3 = 1
 d3 = 0
 a3 = 0
 pa4 - 9 a4 <= 0
 pa4 - l4 <= 0
 pa4 - l4 - 9 a4 >= -9
 pb4 - 9 b4 <= 0
 pb4 - l4 <= 0
 pb4 - l4 - 9 b4 >= -9
 pc4 - 9 c4 <= 0
 pc4 - l4 <= 0
 pc4 - l4 - 9 c4 >= -9
 pd4 - 9 d4 <= 0
 pd4 - l4 <= 0
 pd4 - l4 - 9 d4 >= -9
 vx2 - vx15 - pa4 + pb4 = 0
 vy2 - vy15 - pc4 + pd4 = 0
 a4 + b4 <= 1
 c4 + d4 <= 1
 c4 = 1
 d4 = 0
 a4 = 0
 pa5 - 9 a5 <= 0
 pa5 - l5 <= 0
 pa5 - l5 - 9 a5 >= -9
 pb5 - 9 b5 <= 0
 pb5 - l5 <= 0
 pb5 - l5 - 9 b5 >= -9
 pc5 - 9 c5 <= 0
 pc5 - l5 <= 0
 pc5 - l5 - 9 c5 >= -9
 pd5 - 9 d5 <= 0
 pd5 - l5 <= 0
 pd5 - l5 - 9 d5 >= -9
 vx0 - vx2 - pa5 + pb5 = 0
 vy0 - vy2 - pc5 + pd5 = 0
 a5 + b5 <= 1
 c5 + d5 <= 1
 a5 = 0
 d5 = 0
 b5 = 1
 pa6 - 9 a6 <= 0
 pa6 - l6 <= 0
 pa6 - l6 - 9 a6 >= -9
 pb6 - 9 b6 <= 0
 pb6 - l6 <= 0
 pb6 - l6 - 9 b6 >= -9
 pc6 - 9 c6 <= 0
 pc6 - l6 <= 0
 pc6 - l6 - 9 c6 >= -9
 pd6 - 9 d6 <= 0
 pd6 - l6 <= 0
 pd6 - l6 - 9 d6 >= -9
 vx14 - vx0 - pa6 + pb6 = 0
 vy14 - vy0 - pc6 + pd6 = 0
 a6 + b6 <= 1
 c6 + d6 <= 1
 a6 = 0
 b6 = 1
 d6 = 0
 pa7 - 9 a7 <= 0
 pa7 - l7 <= 0
 pa7 - l7 - 9 a7 >= -9
 pb7 - 9 b7 <= 0
 pb7 - l7 <= 0
 pb7 - l7 - 9 b7 >= -9
 pc7 - 9 c7 <= 0
 pc7 - l7 <= 0
 pc7 - l7 - 9 c7 >= -9
 pd7 - 9 d7 <= 0
 pd7 - l7 <= 0
 pd7 - l7 - 9 d7 >= -9
 vx16 - vx14 - pa7 + pb7 = 0
 vy16 - vy14 - pc7 + pd7 = 0
 a7 + b7 <= 1
 c7 + d7 <= 1
 a7 = 0
 b7 = 1
 d7 = 0
 pa8 - 9 a8 <= 0
 pa8 - l8 <= 0
 pa8 - l8 - 9 a8 >= -9
 pb8 - 9 b8 <= 0
 pb8 - l8 <= 0
 pb8 - l8 - 9 b8 >= -9
 pc8 - 9 c8 <= 0
 pc8 - l8 <= 0
 pc8 - l8 - 9 c8 >= -9
 pd8 - 9 d8 <= 0
 pd8 - l8 <= 0
 pd8 - l8 - 9 d8 >= -9
 vx10 - vx16 - pa8 + pb8 = 0
 vy10 - vy16 - pc8 + pd8 = 0
 a8 + b8 <= 1
 c8 + d8 <= 1
 a8 = 0
 b8 = 1
 d8 = 0
 pa9 - 9 a9 <= 0
 pa9 - l9 <= 0
 pa9 - l9 - 9 a9 >= -9
 pb9 - 9 b9 <= 0
 pb9 - l9 <= 0
 pb9 - l9 - 9 b9 >= -9
 pc9 - 9 c9 <= 0
 pc9 - l9 <= 0
 pc9 - l9 - 9 c9 >= -9
 pd9 - 9 d9 <= 0
 pd9 - l9 <= 0
 pd9 - l9 - 9 d9 >= -9
 vx12 - vx10 - pa9 + pb9 = 0
 vy12 - vy10 - pc9 + pd9 = 0
 a9 + b9 <= 1
 c9 + d9 <= 1
 a9 = 0
 b9 = 1
 d9 = 0
 pa10 - 9 a10 <= 0
 pa10 - l10 <= 0
 pa10 - l10 - 9 a10 >= -9
 pb10 - 9 b10 <= 0
 pb10 - l10 <= 0
 pb10 - l10 - 9 b10 >= -9
 pc10 - 9 c10 <= 0
 pc10 - l10 <= 0
 pc10 - l10 - 9 c10 >= -9
 pd10 - 9 d10 <= 0
 pd10 - l10 <= 0
 pd10 - l10 - 9 d10 >= -9
 vx9 - vx12 - pa10 + pb10 = 0
 vy9 - vy12 - pc10 + pd10 = 0
 a10 + b10 <= 1
 c10 + d10 <= 1
 a10 = 0
 b10 = 1
 d10 = 0
 pa11 - 9 a11 <= 0
 pa11 - l11 <= 0
 pa11 - l11 - 9 a11 >= -9
 pb11 - 9 b11 <= 0
 pb11 - l11 <= 0
 pb11 - l11 - 9 b11 >= -9
 pc11 - 9 c11 <= 0
 pc11 - l11 <= 0
 pc11 - l11 - 9 c11 >= -9
 pd11 - 9 d11 <= 0
 pd11 - l11 <= 0
 pd11 - l11 - 9 d11 >= -9
 vx1 - vx9 - pa11 + pb11 = 0
 vy1 - vy9 - pc11 + pd11 = 0
 a11 + b11 <= 1
 c11 + d11 <= 1
 a11 = 0
 b11 = 1
 d11 = 0
 pa12 - 9 a12 <= 0
 pa12 - l12 <= 0
 pa12 - l12 - 9 a12 >= -9
 pb12 - 9 b12 <= 0
 pb12 - l12 <= 0
 pb12 - l12 - 9 b12 >= -9
 pc12 - 9 c12 <= 0
 pc12 - l12 <= 0
 pc12 - l12 - 9 c12 >= -9
 pd12 - 9 d12 <= 0
 pd12 - l12 <= 0
 pd12 - l12 - 9 d12 >= -9
 vx7 - vx1 - pa12 + pb12 = 0
 vy7 - vy1 - pc12 + pd12 = 0
 a12 + b12 <= 1
 c12 + d12 <= 1
 a12 = 0
 b12 = 1
 d12 = 0
 pa13 - 9 a13 <= 0
 pa13 - l13 <= 0
 pa13 - l13 - 9 a13 >= -9
 pb13 - 9 b13 <= 0
 pb13 - l13 <= 0
 pb13 - l13 - 9 b13 >= -9
 pc13 - 9 c13 <= 0
 pc13 - l13 <= 0
 pc13 - l13 - 9 c13 >= -9
 pd13 - 9 d13 <= 0
 pd13 - l13 <= 0
 pd13 - l13 - 9 d13 >= -9
 vx6 - vx7 - pa13 + pb13 = 0
 vy6 - vy7 - pc13 + pd13 = 0
 a13 + b13 <= 1
 c13 + d13 <= 1
 a13 = 0
 b13 = 1
 d13 = 0
 pa14 - 9 a14 <= 0
 pa14 - l14 <= 0
 pa14 - l14 - 9 a14 >= -9
 pb14 - 9 b14 <= 0
 pb14 - l14 <= 0
 pb14 - l14 - 9 b14 >= -9
 pc14 - 9 c14 <= 0
 pc14 - l14 <= 0
 pc14 - l14 - 9 c14 >= -9
 pd14 - 9 d14 <= 0
 pd14 - l14 <= 0
 pd14 - l14 - 9 d14 >= -9
 vx8 - vx6 - pa14 + pb14 = 0
 vy8 - vy6 - pc14 + pd14 = 0
 a14 + b14 <= 1
 c14 + d14 <= 1
 a14 = 0
 b14 = 1
 d14 = 0
 pa15 - 9 a15 <= 0
 pa15 - l15 <= 0
 pa15 - l15 - 9 a15 >= -9
 pb15 - 9 b15 <= 0
 pb15 - l15 <= 0
 pb15 - l15 - 9 b15 >= -9
 pc15 - 9 c15 <= 0
 pc15 - l15 <= 0
 pc15 - l15 - 9 c15 >= -9
 pd15 - 9 d15 <= 0
 pd15 - l15 <= 0
 pd15 - l15 - 9 d15 >= -9
 vx11 - vx8 - pa15 + pb15 = 0
 vy11 - vy8 - pc15 + pd15 = 0
 a15 + b15 <= 1
 c15 + d15 <= 1
 a15 = 0
 d15 = 0
 b15 = 1
 pa16 - 9 a16 <= 0
 pa16 - l16 <= 0
 pa16 - l16 - 9 a16 >= -9
 pb16 - 9 b16 <= 0
 pb16 - l16 <= 0
 pb16 - l16 - 9 b16 >= -9
 pc16 - 9 c16 <= 0
 pc16 - l16 <= 0
 pc16 - l16 - 9 c16 >= -9
 pd16 - 9 d16 <= 0
 pd16 - l16 <= 0
 pd16 - l16 - 9 d16 >= -9
 vx13 - vx11 - pa16 + pb16 = 0
 vy13 - vy11 - pc16 + pd16 = 0
 a16 + b16 <= 1
 c16 + d16 <= 1
 a16 = 0
 b16 = 1
 d16 = 0
 pa17 - 9 a17 <= 0
 pa17 - l17 <= 0
 pa17 - l17 - 9 a17 >= -9
 pb17 - 9 b17 <= 0
 pb17 - l17 <= 0
 pb17 - l17 - 9 b17 >= -9
 pc17 - 9 c17 <= 0
 pc17 - l17 <= 0
 pc17 - l17 - 9 c17 >= -9
 pd17 - 9 d17 <= 0
 pd17 - l17 <= 0
 pd17 - l17 - 9 d17 >= -9
 vx4 - vx13 - pa17 + pb17 = 0
 vy4 - vy13 - pc17 + pd17 = 0
 a17 + b17 <= 1
 c17 + d17 <= 1
 a17 = 0
 b17 = 1
 d17 = 0
 pa18 - 9 a18 <= 0
 pa18 - l18 <= 0
 pa18 - l18 - 9 a18 >= -9
 pb18 - 9 b18 <= 0
 pb18 - l18 <= 0
 pb18 - l18 - 9 b18 >= -9
 pc18 - 9 c18 <= 0
 pc18 - l18 <= 0
 pc18 - l18 - 9 c18 >= -9
 pd18 - 9 d18 <= 0
 pd18 - l18 <= 0
 pd18 - l18 - 9 d18 >= -9
 vx36 - vx4 - pa18 + pb18 = 0
 vy36 - vy4 - pc18 + pd18 = 0
 a18 + b18 <= 1
 c18 + d18 <= 1
 a18 = 0
 c18 = 0
 d18 = 1
 pa19 - 9 a19 <= 0
 pa19 - l19 <= 0
 pa19 - l19 - 9 a19 >= -9
 pb19 - 9 b19 <= 0
 pb19 - l19 <= 0
 pb19 - l19 - 9 b19 >= -9
 pc19 - 9 c19 <= 0
 pc19 - l19 <= 0
 pc19 - l19 - 9 c19 >= -9
 pd19 - 9 d19 <= 0
 pd19 - l19 <= 0
 pd19 - l19 - 9 d19 >= -9
 vx39 - vx36 - pa19 + pb19 = 0
 vy39 - vy36 - pc19 + pd19 = 0
 a19 + b19 <= 1
 c19 + d19 <= 1
 a19 = 0
 c19 = 0
 b19 = 1
 pa20 - 9 a20 <= 0
 pa20 - l20 <= 0
 pa20 - l20 - 9 a20 >= -9
 pb20 - 9 b20 <= 0
 pb20 - l20 <= 0
 pb20 - l20 - 9 b20 >= -9
 pc20 - 9 c20 <= 0
 pc20 - l20 <= 0
 pc20 - l20 - 9 c20 >= -9
 pd20 - 9 d20 <= 0
 pd20 - l20 <= 0
 pd20 - l20 - 9 d20 >= -9
 vx38 - vx39 - pa20 + pb20 = 0
 vy38 - vy39 - pc20 + pd20 = 0
 a20 + b20 <= 1
 c20 + d20 <= 1
 a20 = 0
 c20 = 0
 b20 = 1
 pa21 - 9 a21 <= 0
 pa21 - l21 <= 0
 pa21 - l21 - 9 a21 >= -9
 pb21 - 9 b21 <= 0
 pb21 - l21 <= 0
 pb21 - l21 - 9 b21 >= -9
 pc21 - 9 c21 <= 0
 pc21 - l21 <= 0
 pc21 - l21 - 9 c21 >= -9
 pd21 - 9 d21 <= 0
 pd21 - l21 <= 0
 pd21 - l21 - 9 d21 >= -9
 vx37 - vx38 - pa21 + pb21 = 0
 vy37 - vy38 - pc21 + pd21 = 0
 a21 + b21 <= 1
 c21 + d21 <= 1
 a21 = 0
 c21 = 0
 b21 = 1
 pa22 - 9 a22 <= 0
 pa22 - l22 <= 0
 pa22 - l22 - 9 a22 >= -9
 pb22 - 9 b22 <= 0
 pb22 - l22 <= 0
 pb22 - l22 - 9 b22 >= -9
 pc22 - 9 c22 <= 0
 pc22 - l22 <= 0
 pc22 - l22 - 9 c22 >= -9
 pd22 - 9 d22 <= 0
 pd22 - l22 <= 0
 pd22 - l22 - 9 d22 >= -9
 vx27 - vx17 - pa22 + pb22 = 0
 vy27 - vy17 - pc22 + pd22 = 0
 a22 + b22 <= 1
 c22 + d22 <= 1
 c22 = 1
 d22 = 0
 a22 = 0
 pa23 - 9 a23 <= 0
 pa23 - l23 <= 0
 pa23 - l23 - 9 a23 >= -9
 pb23 - 9 b23 <= 0
 pb23 - l23 <= 0
 pb23 - l23 - 9 b23 >= -9
 pc23 - 9 c23 <= 0
 pc23 - l23 <= 0
 pc23 - l23 - 9 c23 >= -9
 pd23 - 9 d23 <= 0
 pd23 - l23 <= 0
 pd23 - l23 - 9 d23 >= -9
 vx30 - vx27 - pa23 + pb23 = 0
 vy30 - vy27 - pc23 + pd23 = 0
 a23 + b23 <= 1
 c23 + d23 <= 1
 c23 = 1
 d23 = 0
 a23 = 0
 pa24 - 9 a24 <= 0
 pa24 - l24 <= 0
 pa24 - l24 - 9 a24 >= -9
 pb24 - 9 b24 <= 0
 pb24 - l24 <= 0
 pb24 - l24 - 9 b24 >= -9
 pc24 - 9 c24 <= 0
 pc24 - l24 <= 0
 pc24 - l24 - 9 c24 >= -9
 pd24 - 9 d24 <= 0
 pd24 - l24 <= 0
 pd24 - l24 - 9 d24 >= -9
 vx31 - vx30 - pa24 + pb24 = 0
 vy31 - vy30 - pc24 + pd24 = 0
 a24 + b24 <= 1
 c24 + d24 <= 1
 c24 = 1
 d24 = 0
 a24 = 0
 pa25 - 9 a25 <= 0
 pa25 - l25 <= 0
 pa25 - l25 - 9 a25 >= -9
 pb25 - 9 b25 <= 0
 pb25 - l25 <= 0
 pb25 - l25 - 9 b25 >= -9
 pc25 - 9 c25 <= 0
 pc25 - l25 <= 0
 pc25 - l25 - 9 c25 >= -9
 pd25 - 9 d25 <= 0
 pd25 - l25 <= 0
 pd25 - l25 - 9 d25 >= -9
 vx33 - vx31 - pa25 + pb25 = 0
 vy33 - vy31 - pc25 + pd25 = 0
 a25 + b25 <= 1
 c25 + d25 <= 1
 c25 = 1
 d25 = 0
 a25 = 0
 pa26 - 9 a26 <= 0
 pa26 - l26 <= 0
 pa26 - l26 - 9 a26 >= -9
 pb26 - 9 b26 <= 0
 pb26 - l26 <= 0
 pb26 - l26 - 9 b26 >= -9
 pc26 - 9 c26 <= 0
 pc26 - l26 <= 0
 pc26 - l26 - 9 c26 >= -9
 pd26 - 9 d26 <= 0
 pd26 - l26 <= 0
 pd26 - l26 - 9 d26 >= -9
 vx29 - vx33 - pa26 + pb26 = 0
 vy29 - vy33 - pc26 + pd26 = 0
 a26 + b26 <= 1
 c26 + d26 <= 1
 c26 = 1
 d26 = 0
 a26 = 0
 pa27 - 9 a27 <= 0
 pa27 - l27 <= 0
 pa27 - l27 - 9 a27 >= -9
 pb27 - 9 b27 <= 0
 pb27 - l27 <= 0
 pb27 - l27 - 9 b27 >= -9
 pc27 - 9 c27 <= 0
 pc27 - l27 <= 0
 pc27 - l27 - 9 c27 >= -9
 pd27 - 9 d27 <= 0
 pd27 - l27 <= 0
 pd27 - l27 - 9 d27 >= -9
 vx34 - vx29 - pa27 + pb27 = 0
 vy34 - vy29 - pc27 + pd27 = 0
 a27 + b27 <= 1
 c27 + d27 <= 1
 c27 = 1
 d27 = 0
 b27 = 0
 pa28 - 9 a28 <= 0
 pa28 - l28 <= 0
 pa28 - l28 - 9 a28 >= -9
 pb28 - 9 b28 <= 0
 pb28 - l28 <= 0
 pb28 - l28 - 9 b28 >= -9
 pc28 - 9 c28 <= 0
 pc28 - l28 <= 0
 pc28 - l28 - 9 c28 >= -9
 pd28 - 9 d28 <= 0
 pd28 - l28 <= 0
 pd28 - l28 - 9 d28 >= -9
 vx32 - vx34 - pa28 + pb28 = 0
 vy32 - vy34 - pc28 + pd28 = 0
 a28 + b28 <= 1
 c28 + d28 <= 1
 c28 = 1
 d28 = 0
 a28 = 0
 pa29 - 9 a29 <= 0
 pa29 - l29 <= 0
 pa29 - l29 - 9 a29 >= -9
 pb29 - 9 b29 <= 0
 pb29 - l29 <= 0
 pb29 - l29 - 9 b29 >= -9
 pc29 - 9 c29 <= 0
 pc29 - l29 <= 0
 pc29 - l29 - 9 c29 >= -9
 pd29 - 9 d29 <= 0
 pd29 - l29 <= 0
 pd29 - l29 - 9 d29 >= -9
 vx65 - vx32 - pa29 + pb29 = 0
 vy65 - vy32 - pc29 + pd29 = 0
 a29 + b29 <= 1
 c29 + d29 <= 1
 c29 = 1
 d29 = 0
 b29 = 0
 pa30 - 9 a30 <= 0
 pa30 - l30 <= 0
 pa30 - l30 - 9 a30 >= -9
 pb30 - 9 b30 <= 0
 pb30 - l30 <= 0
 pb30 - l30 - 9 b30 >= -9
 pc30 - 9 c30 <= 0
 pc30 - l30 <= 0
 pc30 - l30 - 9 c30 >= -9
 pd30 - 9 d30 <= 0
 pd30 - l30 <= 0
 pd30 - l30 - 9 d30 >= -9
 vx28 - vx65 - pa30 + pb30 = 0
 vy28 - vy65 - pc30 + pd30 = 0
 a30 + b30 <= 1
 c30 + d30 <= 1
 c30 = 1
 d30 = 0
 a30 = 0
 pa31 - 9 a31 <= 0
 pa31 - l31 <= 0
 pa31 - l31 - 9 a31 >= -9
 pb31 - 9 b31 <= 0
 pb31 - l31 <= 0
 pb31 - l31 - 9 b31 >= -9
 pc31 - 9 c31 <= 0
 pc31 - l31 <= 0
 pc31 - l31 - 9 c31 >= -9
 pd31 - 9 d31 <= 0
 pd31 - l31 <= 0
 pd31 - l31 - 9 d31 >= -9
 vx18 - vx28 - pa31 + pb31 = 0
 vy18 - vy28 - pc31 + pd31 = 0
 a31 + b31 <= 1
 c31 + d31 <= 1
 a31 = 0
 d31 = 0
 c31 = 1
 pa32 - 9 a32 <= 0
 pa32 - l32 <= 0
 pa32 - l32 - 9 a32 >= -9
 pb32 - 9 b32 <= 0
 pb32 - l32 <= 0
 pb32 - l32 - 9 b32 >= -9
 pc32 - 9 c32 <= 0
 pc32 - l32 <= 0
 pc32 - l32 - 9 c32 >= -9
 pd32 - 9 d32 <= 0
 pd32 - l32 <= 0
 pd32 - l32 - 9 d32 >= -9
 vx19 - vx18 - pa32 + pb32 = 0
 vy19 - vy18 - pc32 + pd32 = 0
 a32 + b32 <= 1
 c32 + d32 <= 1
 a32 = 0
 d32 = 0
 c32 = 1
 pa33 - 9 a33 <= 0
 pa33 - l33 <= 0
 pa33 - l33 - 9 a33 >= -9
 pb33 - 9 b33 <= 0
 pb33 - l33 <= 0
 pb33 - l33 - 9 b33 >= -9
 pc33 - 9 c33 <= 0
 pc33 - l33 <= 0
 pc33 - l33 - 9 c33 >= -9
 pd33 - 9 d33 <= 0
 pd33 - l33 <= 0
 pd33 - l33 - 9 d33 >= -9
 vx26 - vx19 - pa33 + pb33 = 0
 vy26 - vy19 - pc33 + pd33 = 0
 a33 + b33 <= 1
 c33 + d33 <= 1
 c33 = 1
 d33 = 0
 b33 = 0
 pa34 - 9 a34 <= 0
 pa34 - l34 <= 0
 pa34 - l34 - 9 a34 >= -9
 pb34 - 9 b34 <= 0
 pb34 - l34 <= 0
 pb34 - l34 - 9 b34 >= -9
 pc34 - 9 c34 <= 0
 pc34 - l34 <= 0
 pc34 - l34 - 9 c34 >= -9
 pd34 - 9 d34 <= 0
 pd34 - l34 <= 0
 pd34 - l34 - 9 d34 >= -9
 vx21 - vx26 - pa34 + pb34 = 0
 vy21 - vy26 - pc34 + pd34 = 0
 a34 + b34 <= 1
 c34 + d34 <= 1
 c34 = 1
 d34 = 0
 b34 = 0
 pa35 - 9 a35 <= 0
 pa35 - l35 <= 0
 pa35 - l35 - 9 a35 >= -9
 pb35 - 9 b35 <= 0
 pb35 - l35 <= 0
 pb35 - l35 - 9 b35 >= -9
 pc35 - 9 c35 <= 0
 pc35 - l35 <= 0
 pc35 - l35 - 9 c35 >= -9
 pd35 - 9 d35 <= 0
 pd35 - l35 <= 0
 pd35 - l35 - 9 d35 >= -9
 vx24 - vx21 - pa35 + pb35 = 0
 vy24 - vy21 - pc35 + pd35 = 0
 a35 + b35 <= 1
 c35 + d35 <= 1
 c35 = 1
 d35 = 0
 b35 = 0
 pa36 - 9 a36 <= 0
 pa36 - l36 <= 0
 pa36 - l36 - 9 a36 >= -9
 pb36 - 9 b36 <= 0
 pb36 - l36 <= 0
 pb36 - l36 - 9 b36 >= -9
 pc36 - 9 c36 <= 0
 pc36 - l36 <= 0
 pc36 - l36 - 9 c36 >= -9
 pd36 - 9 d36 <= 0
 pd36 - l36 <= 0
 pd36 - l36 - 9 d36 >= -9
 vx23 - vx24 - pa36 + pb36 = 0
 vy23 - vy24 - pc36 + pd36 = 0
 a36 + b36 <= 1
 c36 + d36 <= 1
 c36 = 1
 d36 = 0
 b36 = 0
 pa37 - 9 a37 <= 0
 pa37 - l37 <= 0
 pa37 - l37 - 9 a37 >= -9
 pb37 - 9 b37 <= 0
 pb37 - l37 <= 0
 pb37 - l37 - 9 b37 >= -9
 pc37 - 9 c37 <= 0
 pc37 - l37 <= 0
 pc37 - l37 - 9 c37 >= -9
 pd37 - 9 d37 <= 0
 pd37 - l37 <= 0
 pd37 - l37 - 9 d37 >= -9
 vx20 - vx23 - pa37 + pb37 = 0
 vy20 - vy23 - pc37 + pd37 = 0
 a37 + b37 <= 1
 c37 + d37 <= 1
 c37 = 1
 d37 = 0
 b37 = 0
 pa38 - 9 a38 <= 0
 pa38 - l38 <= 0
 pa38 - l38 - 9 a38 >= -9
 pb38 - 9 b38 <= 0
 pb38 - l38 <= 0
 pb38 - l38 - 9 b38 >= -9
 pc38 - 9 c38 <= 0
 pc38 - l38 <= 0
 pc38 - l38 - 9 c38 >= -9
 pd38 - 9 d38 <= 0
 pd38 - l38 <= 0
 pd38 - l38 - 9 d38 >= -9
 vx25 - vx20 - pa38 + pb38 = 0
 vy25 - vy20 - pc38 + pd38 = 0
 a38 + b38 <= 1
 c38 + d38 <= 1
 c38 = 1
 d38 = 0
 b38 = 0
 pa39 - 9 a39 <= 0
 pa39 - l39 <= 0
 pa39 - l39 - 9 a39 >= -9
 pb39 - 9 b39 <= 0
 pb39 - l39 <= 0
 pb39 - l39 - 9 b39 >= -9
 pc39 - 9 c39 <= 0
 pc39 - l39 <= 0
 pc39 - l39 - 9 c39 >= -9
 pd39 - 9 d39 <= 0
 pd39 - l39 <= 0
 pd39 - l39 - 9 d39 >= -9
 vx22 - vx25 - pa39 + pb39 = 0
 vy22 - vy25 - pc39 + pd39 = 0
 a39 + b39 <= 1
 c39 + d39 <= 1
 c39 = 1
 d39 = 0
 a39 = 0
 pa40 - 9 a40 <= 0
 pa40 - l40 <= 0
 pa40 - l40 - 9 a40 >= -9
 pb40 - 9 b40 <= 0
 pb40 - l40 <= 0
 pb40 - l40 - 9 b40 >= -9
 pc40 - 9 c40 <= 0
 pc40 - l40 <= 0
 pc40 - l40 - 9 c40 >= -9
 pd40 - 9 d40 <= 0
 pd40 - l40 <= 0
 pd40 - l40 - 9 d40 >= -9
 vx64 - vx41 - pa40 + pb40 = 0
 vy64 - vy41 - pc40 + pd40 = 0
 a40 + b40 <= 1
 c40 + d40 <= 1
 a40 = 0
 d40 = 0
 b40 = 1
 pa41 - 9 a41 <= 0
 pa41 - l41 <= 0
 pa41 - l41 - 9 a41 >= -9
 pb41 - 9 b41 <= 0
 pb41 - l41 <= 0
 pb41 - l41 - 9 b41 >= -9
 pc41 - 9 c41 <= 0
 pc41 - l41 <= 0
 pc41 - l41 - 9 c41 >= -9
 pd41 - 9 d41 <= 0
 pd41 - l41 <= 0
 pd41 - l41 - 9 d41 >= -9
 vx43 - vx64 - pa41 + pb41 = 0
 vy43 - vy64 - pc41 + pd41 = 0
 a41 + b41 <= 1
 c41 + d41 <= 1
 a41 = 0
 b41 = 1
 d41 = 0
 pa42 - 9 a42 <= 0
 pa42 - l42 <= 0
 pa42 - l42 - 9 a42 >= -9
 pb42 - 9 b42 <= 0
 pb42 - l42 <= 0
 pb42 - l42 - 9 b42 >= -9
 pc42 - 9 c42 <= 0
 pc42 - l42 <= 0
 pc42 - l42 - 9 c42 >= -9
 pd42 - 9 d42 <= 0
 pd42 - l42 <= 0
 pd42 - l42 - 9 d42 >= -9
 vx44 - vx43 - pa42 + pb42 = 0
 vy44 - vy43 - pc42 + pd42 = 0
 a42 + b42 <= 1
 c42 + d42 <= 1
 a42 = 0
 d42 = 0
 c42 = 1
 pa43 - 9 a43 <= 0
 pa43 - l43 <= 0
 pa43 - l43 - 9 a43 >= -9
 pb43 - 9 b43 <= 0
 pb43 - l43 <= 0
 pb43 - l43 - 9 b43 >= -9
 pc43 - 9 c43 <= 0
 pc43 - l43 <= 0
 pc43 - l43 - 9 c43 >= -9
 pd43 - 9 d43 <= 0
 pd43 - l43 <= 0
 pd43 - l43 - 9 d43 >= -9
 vx3 - vx44 - pa43 + pb43 = 0
 vy3 - vy44 - pc43 + pd43 = 0
 a43 + b43 <= 1
 c43 + d43 <= 1
 a43 = 0
 b43 = 1
 d43 = 0
 pa44 - 9 a44 <= 0
 pa44 - l44 <= 0
 pa44 - l44 - 9 a44 >= -9
 pb44 - 9 b44 <= 0
 pb44 - l44 <= 0
 pb44 - l44 - 9 b44 >= -9
 pc44 - 9 c44 <= 0
 pc44 - l44 <= 0
 pc44 - l44 - 9 c44 >= -9
 pd44 - 9 d44 <= 0
 pd44 - l44 <= 0
 pd44 - l44 - 9 d44 >= -9
 vx45 - vx3 - pa44 + pb44 = 0
 vy45 - vy3 - pc44 + pd44 = 0
 a44 + b44 <= 1
 c44 + d44 <= 1
 a44 = 0
 b44 = 1
 c44 = 0
 pa45 - 9 a45 <= 0
 pa45 - l45 <= 0
 pa45 - l45 - 9 a45 >= -9
 pb45 - 9 b45 <= 0
 pb45 - l45 <= 0
 pb45 - l45 - 9 b45 >= -9
 pc45 - 9 c45 <= 0
 pc45 - l45 <= 0
 pc45 - l45 - 9 c45 >= -9
 pd45 - 9 d45 <= 0
 pd45 - l45 <= 0
 pd45 - l45 - 9 d45 >= -9
 vx46 - vx45 - pa45 + pb45 = 0
 vy46 - vy45 - pc45 + pd45 = 0
 a45 + b45 <= 1
 c45 + d45 <= 1
 a45 = 0
 b45 = 1
 c45 = 0
 pa46 - 9 a46 <= 0
 pa46 - l46 <= 0
 pa46 - l46 - 9 a46 >= -9
 pb46 - 9 b46 <= 0
 pb46 - l46 <= 0
 pb46 - l46 - 9 b46 >= -9
 pc46 - 9 c46 <= 0
 pc46 - l46 <= 0
 pc46 - l46 - 9 c46 >= -9
 pd46 - 9 d46 <= 0
 pd46 - l46 <= 0
 pd46 - l46 - 9 d46 >= -9
 vx47 - vx46 - pa46 + pb46 = 0
 vy47 - vy46 - pc46 + pd46 = 0
 a46 + b46 <= 1
 c46 + d46 <= 1
 a46 = 0
 b46 = 1
 d46 = 0
 pa47 - 9 a47 <= 0
 pa47 - l47 <= 0
 pa47 - l47 - 9 a47 >= -9
 pb47 - 9 b47 <= 0
 pb47 - l47 <= 0
 pb47 - l47 - 9 b47 >= -9
 pc47 - 9 c47 <= 0
 pc47 - l47 <= 0
 pc47 - l47 - 9 c47 >= -9
 pd47 - 9 d47 <= 0
 pd47 - l47 <= 0
 pd47 - l47 - 9 d47 >= -9
 vx48 - vx47 - pa47 + pb47 = 0
 vy48 - vy47 - pc47 + pd47 = 0
 a47 + b47 <= 1
 c47 + d47 <= 1
 a47 = 0
 b47 = 1
 d47 = 0
 pa48 - 9 a48 <= 0
 pa48 - l48 <= 0
 pa48 - l48 - 9 a48 >= -9
 pb48 - 9 b48 <= 0
 pb48 - l48 <= 0
 pb48 - l48 - 9 b48 >= -9
 pc48 - 9 c48 <= 0
 pc48 - l48 <= 0
 pc48 - l48 - 9 c48 >= -9
 pd48 - 9 d48 <= 0
 pd48 - l48 <= 0
 pd48 - l48 - 9 d48 >= -9
 vx35 - vx48 - pa48 + pb48 = 0
 vy35 - vy48 - pc48 + pd48 = 0
 a48 + b48 <= 1
 c48 + d48 <= 1
 a48 = 0
 b48 = 1
 c48 = 0
 pa49 - 9 a49 <= 0
 pa49 - l49 <= 0
 pa49 - l49 - 9 a49 >= -9
 pb49 - 9 b49 <= 0
 pb49 - l49 <= 0
 pb49 - l49 - 9 b49 >= -9
 pc49 - 9 c49 <= 0
 pc49 - l49 <= 0
 pc49 - l49 - 9 c49 >= -9
 pd49 - 9 d49 <= 0
 pd49 - l49 <= 0
 pd49 - l49 - 9 d49 >= -9
 vx49 - vx35 - pa49 + pb49 = 0
 vy49 - vy35 - pc49 + pd49 = 0
 a49 + b49 <= 1
 c49 + d49 <= 1
 a49 = 0
 b49 = 1
 d49 = 0
 pa50 - 9 a50 <= 0
 pa50 - l50 <= 0
 pa50 - l50 - 9 a50 >= -9
 pb50 - 9 b50 <= 0
 pb50 - l50 <= 0
 pb50 - l50 - 9 b50 >= -9
 pc50 - 9 c50 <= 0
 pc50 - l50 <= 0
 pc50 - l50 - 9 c50 >= -9
 pd50 - 9 d50 <= 0
 pd50 - l50 <= 0
 pd50 - l50 - 9 d50 >= -9
 vx50 - vx49 - pa50 + pb50 = 0
 vy50 - vy49 - pc50 + pd50 = 0
 a50 + b50 <= 1
 c50 + d50 <= 1
 a50 = 0
 b50 = 1
 d50 = 0
 pa51 - 9 a51 <= 0
 pa51 - l51 <= 0
 pa51 - l51 - 9 a51 >= -9
 pb51 - 9 b51 <= 0
 pb51 - l51 <= 0
 pb51 - l51 - 9 b51 >= -9
 pc51 - 9 c51 <= 0
 pc51 - l51 <= 0
 pc51 - l51 - 9 c51 >= -9
 pd51 - 9 d51 <= 0
 pd51 - l51 <= 0
 pd51 - l51 - 9 d51 >= -9
 vx51 - vx50 - pa51 + pb51 = 0
 vy51 - vy50 - pc51 + pd51 = 0
 a51 + b51 <= 1
 c51 + d51 <= 1
 c51 = 1
 d51 = 0
 a51 = 0
 pa52 - 9 a52 <= 0
 pa52 - l52 <= 0
 pa52 - l52 - 9 a52 >= -9
 pb52 - 9 b52 <= 0
 pb52 - l52 <= 0
 pb52 - l52 - 9 b52 >= -9
 pc52 - 9 c52 <= 0
 pc52 - l52 <= 0
 pc52 - l52 - 9 c52 >= -9
 pd52 - 9 d52 <= 0
 pd52 - l52 <= 0
 pd52 - l52 - 9 d52 >= -9
 vx52 - vx51 - pa52 + pb52 = 0
 vy52 - vy51 - pc52 + pd52 = 0
 a52 + b52 <= 1
 c52 + d52 <= 1
 c52 = 1
 d52 = 0
 b52 = 0
 pa53 - 9 a53 <= 0
 pa53 - l53 <= 0
 pa53 - l53 - 9 a53 >= -9
 pb53 - 9 b53 <= 0
 pb53 - l53 <= 0
 pb53 - l53 - 9 b53 >= -9
 pc53 - 9 c53 <= 0
 pc53 - l53 <= 0
 pc53 - l53 - 9 c53 >= -9
 pd53 - 9 d53 <= 0
 pd53 - l53 <= 0
 pd53 - l53 - 9 d53 >= -9
 vx62 - vx52 - pa53 + pb53 = 0
 vy62 - vy52 - pc53 + pd53 = 0
 a53 + b53 <= 1
 c53 + d53 <= 1
 c53 = 1
 d53 = 0
 b53 = 0
 pa54 - 9 a54 <= 0
 pa54 - l54 <= 0
 pa54 - l54 - 9 a54 >= -9
 pb54 - 9 b54 <= 0
 pb54 - l54 <= 0
 pb54 - l54 - 9 b54 >= -9
 pc54 - 9 c54 <= 0
 pc54 - l54 <= 0
 pc54 - l54 - 9 c54 >= -9
 pd54 - 9 d54 <= 0
 pd54 - l54 <= 0
 pd54 - l54 - 9 d54 >= -9
 vx53 - vx62 - pa54 + pb54 = 0
 vy53 - vy62 - pc54 + pd54 = 0
 a54 + b54 <= 1
 c54 + d54 <= 1
 b54 = 0
 d54 = 0
 c54 = 1
 pa55 - 9 a55 <= 0
 pa55 - l55 <= 0
 pa55 - l55 - 9 a55 >= -9
 pb55 - 9 b55 <= 0
 pb55 - l55 <= 0
 pb55 - l55 - 9 b55 >= -9
 pc55 - 9 c55 <= 0
 pc55 - l55 <= 0
 pc55 - l55 - 9 c55 >= -9
 pd55 - 9 d55 <= 0
 pd55 - l55 <= 0
 pd55 - l55 - 9 d55 >= -9
 vx54 - vx53 - pa55 + pb55 = 0
 vy54 - vy53 - pc55 + pd55 = 0
 a55 + b55 <= 1
 c55 + d55 <= 1
 c55 = 1
 d55 = 0
 a55 = 0
 pa56 - 9 a56 <= 0
 pa56 - l56 <= 0
 pa56 - l56 - 9 a56 >= -9
 pb56 - 9 b56 <= 0
 pb56 - l56 <= 0
 pb56 - l56 - 9 b56 >= -9
 pc56 - 9 c56 <= 0
 pc56 - l56 <= 0
 pc56 - l56 - 9 c56 >= -9
 pd56 - 9 d56 <= 0
 pd56 - l56 <= 0
 pd56 - l56 - 9 d56 >= -9
 vx55 - vx54 - pa56 + pb56 = 0
 vy55 - vy54 - pc56 + pd56 = 0
 a56 + b56 <= 1
 c56 + d56 <= 1
 c56 = 1
 d56 = 0
 a56 = 0
 pa57 - 9 a57 <= 0
 pa57 - l57 <= 0
 pa57 - l57 - 9 a57 >= -9
 pb57 - 9 b57 <= 0
 pb57 - l57 <= 0
 pb57 - l57 - 9 b57 >= -9
 pc57 - 9 c57 <= 0
 pc57 - l57 <= 0
 pc57 - l57 - 9 c57 >= -9
 pd57 - 9 d57 <= 0
 pd57 - l57 <= 0
 pd57 - l57 - 9 d57 >= -9
 vx56 - vx55 - pa57 + pb57 = 0
 vy56 - vy55 - pc57 + pd57 = 0
 a57 + b57 <= 1
 c57 + d57 <= 1
 b57 = 0
 d57 = 0
 c57 = 1
 pa58 - 9 a58 <= 0
 pa58 - l58 <= 0
 pa58 - l58 - 9 a58 >= -9
 pb58 - 9 b58 <= 0
 pb58 - l58 <= 0
 pb58 - l58 - 9 b58 >= -9
 pc58 - 9 c58 <= 0
 pc58 - l58 <= 0
 pc58 - l58 - 9 c58 >= -9
 pd58 - 9 d58 <= 0
 pd58 - l58 <= 0
 pd58 - l58 - 9 d58 >= -9
 vx57 - vx56 - pa58 + pb58 = 0
 vy57 - vy56 - pc58 + pd58 = 0
 a58 + b58 <= 1
 c58 + d58 <= 1
 c58 = 1
 d58 = 0
 a58 = 0
 pa59 - 9 a59 <= 0
 pa59 - l59 <= 0
 pa59 - l59 - 9 a59 >= -9
 pb59 - 9 b59 <= 0
 pb59 - l59 <= 0
 pb59 - l59 - 9 b59 >= -9
 pc59 - 9 c59 <= 0
 pc59 - l59 <= 0
 pc59 - l59 - 9 c59 >= -9
 pd59 - 9 d59 <= 0
 pd59 - l59 <= 0
 pd59 - l59 - 9 d59 >= -9
 vx58 - vx57 - pa59 + pb59 = 0
 vy58 - vy57 - pc59 + pd59 = 0
 a59 + b59 <= 1
 c59 + d59 <= 1
 a59 = 0
 d59 = 0
 c59 = 1
 pa60 - 9 a60 <= 0
 pa60 - l60 <= 0
 pa60 - l60 - 9 a60 >= -9
 pb60 - 9 b60 <= 0
 pb60 - l60 <= 0
 pb60 - l60 - 9 b60 >= -9
 pc60 - 9 c60 <= 0
 pc60 - l60 <= 0
 pc60 - l60 - 9 c60 >= -9
 pd60 - 9 d60 <= 0
 pd60 - l60 <= 0
 pd60 - l60 - 9 d60 >= -9
 vx59 - vx58 - pa60 + pb60 = 0
 vy59 - vy58 - pc60 + pd60 = 0
 a60 + b60 <= 1
 c60 + d60 <= 1
 a60 = 0
 d60 = 0
 b60 = 1
 pa61 - 9 a61 <= 0
 pa61 - l61 <= 0
 pa61 - l61 - 9 a61 >= -9
 pb61 - 9 b61 <= 0
 pb61 - l61 <= 0
 pb61 - l61 - 9 b61 >= -9
 pc61 - 9 c61 <= 0
 pc61 - l61 <= 0
 pc61 - l61 - 9 c61 >= -9
 pd61 - 9 d61 <= 0
 pd61 - l61 <= 0
 pd61 - l61 - 9 d61 >= -9
 vx60 - vx59 - pa61 + pb61 = 0
 vy60 - vy59 - pc61 + pd61 = 0
 a61 + b61 <= 1
 c61 + d61 <= 1
 a61 = 0
 d61 = 0
 c61 = 1
 pa62 - 9 a62 <= 0
 pa62 - l62 <= 0
 pa62 - l62 - 9 a62 >= -9
 pb62 - 9 b62 <= 0
 pb62 - l62 <= 0
 pb62 - l62 - 9 b62 >= -9
 pc62 - 9 c62 <= 0
 pc62 - l62 <= 0
 pc62 - l62 - 9 c62 >= -9
 pd62 - 9 d62 <= 0
 pd62 - l62 <= 0
 pd62 - l62 - 9 d62 >= -9
 vx61 - vx60 - pa62 + pb62 = 0
 vy61 - vy60 - pc62 + pd62 = 0
 a62 + b62 <= 1
 c62 + d62 <= 1
 a62 = 0
 d62 = 0
 c62 = 1
 q0 - oa0 - ob0 - oc0 - od0 = 0
 a0 + a1 - 2 ua0 - oa0 = 0
 b0 + b1 - 2 ub0 - ob0 = 0
 c0 + c1 - 2 uc0 - oc0 = 0
 d0 + d1 - 2 ud0 - od0 = 0
 vx40 + vy40 - vx5 - vy5 >= 1
 vx40 + vy40 - vx63 - vy63 >= 1
 vx42 + vy42 - vx5 - vy5 >= 1
 vx42 + vy42 - vx63 - vy63 >= 1
 vx40 - vy40 - vx63 + vy63 >= 1
 vx40 - vy40 - vx15 + vy15 >= 1
 vx42 - vy42 - vx63 + vy63 >= 1
 vx42 - vy42 - vx15 + vy15 >= 1
 vx40 - vy40 - vx15 + vy15 >= 1
 vx40 - vy40 - vx2 + vy2 >= 1
 vx42 - vy42 - vx15 + vy15 >= 1
 vx42 - vy42 - vx2 + vy2 >= 1
 vx40 - vy40 - vx2 + vy2 >= 1
 vx40 - vy40 - vx0 + vy0 >= 1
 vx42 - vy42 - vx2 + vy2 >= 1
 vx42 - vy42 - vx0 + vy0 >= 1
 vx40 - vy40 - vx0 + vy0 >= 1
 vx40 - vy40 - vx14 + vy14 >= 1
 vx42 - vy42 - vx0 + vy0 >= 1
 vx42 - vy42 - vx14 + vy14 >= 1
 vx40 - vy40 - vx14 + vy14 >= 1
 vx40 - vy40 - vx16 + vy16 >= 1
 vx42 - vy42 - vx14 + vy14 >= 1
 vx42 - vy42 - vx16 + vy16 >= 1
 vx40 - vy40 - vx16 + vy16 >= 1
 vx40 - vy40 - vx10 + vy10 >= 1
 vx42 - vy42 - vx16 + vy16 >= 1
 vx42 - vy42 - vx10 + vy10 >= 1
 vx40 - vy40 - vx10 + vy10 >= 1
 vx40 - vy40 - vx12 + vy12 >= 1
 vx42 - vy42 - vx10 + vy10 >= 1
 vx42 - vy42 - vx12 + vy12 >= 1
 vx40 - vy40 - vx12 + vy12 >= 1
 vx40 - vy40 - vx9 + vy9 >= 1
 vx42 - vy42 - vx12 + vy12 >= 1
 vx42 - vy42 - vx9 + vy9 >= 1
 vx40 - vy40 - vx9 + vy9 >= 1
 vx40 - vy40 - vx1 + vy1 >= 1
 vx42 - vy42 - vx9 + vy9 >= 1
 vx42 - vy42 - vx1 + vy1 >= 1
 vx40 - vy40 - vx1 + vy1 >= 1
 vx40 - vy40 - vx7 + vy7 >= 1
 vx42 - vy42 - vx1 + vy1 >= 1
 vx42 - vy42 - vx7 + vy7 >= 1
 vx40 - vy40 - vx7 + vy7 >= 1
 vx40 - vy40 - vx6 + vy6 >= 1
 vx42 - vy42 - vx7 + vy7 >= 1
 vx42 - vy42 - vx6 + vy6 >= 1
 vx40 - vy40 - vx6 + vy6 >= 1
 vx40 - vy40 - vx8 + vy8 >= 1
 vx42 - vy42 - vx6 + vy6 >= 1
 vx42 - vy42 - vx8 + vy8 >= 1
 vx40 - vy40 - vx8 + vy8 >= 1
 vx40 - vy40 - vx11 + vy11 >= 1
 vx42 - vy42 - vx8 + vy8 >= 1
 vx42 - vy42 - vx11 + vy11 >= 1
 vx40 - vy40 - vx11 + vy11 >= 1
 vx40 - vy40 - vx13 + vy13 >= 1
 vx42 - vy42 - vx11 + vy11 >= 1
 vx42 - vy42 - vx13 + vy13 >= 1
 vx40 - vy40 - vx13 + vy13 >= 1
 vx40 - vy40 - vx4 + vy4 >= 1
 vx42 - vy42 - vx13 + vy13 >= 1
 vx42 - vy42 - vx4 + vy4 >= 1
 vx40 - vy40 - vx4 + vy4 >= 1
 vx40 - vy40 - vx36 + vy36 >= 1
 vx42 - vy42 - vx4 + vy4 >= 1
 vx42 - vy42 - vx36 + vy36 >= 1
 vx40 - vy40 - vx36 + vy36 >= 1
 vx40 - vy40 - vx39 + vy39 >= 1
 vx42 - vy42 - vx36 + vy36 >= 1
 vx42 - vy42 - vx39 + vy39 >= 1
 vx40 - vy40 - vx39 + vy39 >= 1
 vx40 - vy40 - vx38 + vy38 >= 1
 vx42 - vy42 - vx39 + vy39 >= 1
 vx42 - vy42 - vx38 + vy38 >= 1
 vx40 - vy40 - vx38 + vy38 >= 1
 vx40 - vy40 - vx37 + vy37 >= 1
 vx42 - vy42 - vx38 + vy38 >= 1
 vx42 - vy42 - vx37 + vy37 >= 1
 vx40 + vy40 - vx17 - vy17 >= 1
 vx40 + vy40 - vx27 - vy27 >= 1
 vx42 + vy42 - vx17 - vy17 >= 1
 vx42 + vy42 - vx27 - vy27 >= 1
 vx40 + vy40 - vx27 - vy27 >= 1
 vx40 + vy40 - vx30 - vy30 >= 1
 vx42 + vy42 - vx27 - vy27 >= 1
 vx42 + vy42 - vx30 - vy30 >= 1
 vx40 + vy40 - vx30 - vy30 >= 1
 vx40 + vy40 - vx31 - vy31 >= 1
 vx42 + vy42 - vx30 - vy30 >= 1
 vx42 + vy42 - vx31 - vy31 >= 1
 vx40 + vy40 - vx31 - vy31 >= 1
 vx40 + vy40 - vx33 - vy33 >= 1
 vx42 + vy42 - vx31 - vy31 >= 1
 vx42 + vy42 - vx33 - vy33 >= 1
 vx40 + vy40 - vx33 - vy33 >= 1
 vx40 + vy40 - vx29 - vy29 >= 1
 vx42 + vy42 - vx33 - vy33 >= 1
 vx42 + vy42 - vx29 - vy29 >= 1
 vx40 + vy40 - vx29 - vy29 >= 1
 vx40 + vy40 - vx34 - vy34 >= 1
 vx42 + vy42 - vx29 - vy29 >= 1
 vx42 + vy42 - vx34 - vy34 >= 1
 vx40 - vx34 >= 1
 vx40 - vx32 >= 1
 vx42 - vx34 >= 1
 vx42 - vx32 >= 1
 vx40 - vy40 - vx32 + vy32 >= 1
 vx40 - vy40 - vx65 + vy65 >= 1
 vx42 - vy42 - vx32 + vy32 >= 1
 vx42 - vy42 - vx65 + vy65 >= 1
 vx40 - vy40 - vx65 + vy65 >= 1
 vx40 - vy40 - vx28 + vy28 >= 1
 vx42 - vy42 - vx65 + vy65 >= 1
 vx42 - vy42 - vx28 + vy28 >= 1
 vx40 - vy40 - vx28 + vy28 >= 1
 vx40 - vy40 - vx18 + vy18 >= 1
 vx42 - vy42 - vx28 + vy28 >= 1
 vx42 - vy42 - vx18 + vy18 >= 1
 vx40 - vy40 - vx18 + vy18 >= 1
 vx40 - vy40 - vx19 + vy19 >= 1
 vx42 - vy42 - vx18 + vy18 >= 1
 vx42 - vy42 - vx19 + vy19 >= 1
 vx40 - vy40 - vx19 + vy19 >= 1
 vx40 - vy40 - vx26 + vy26 >= 1
 vx42 - vy42 - vx19 + vy19 >= 1
 vx42 - vy42 - vx26 + vy26 >= 1
 vx40 - vy40 - vx26 + vy26 >= 1
 vx40 - vy40 - vx21 + vy21 >= 1
 vx42 - vy42 - vx26 + vy26 >= 1
 vx42 - vy42 - vx21 + vy21 >= 1
 vx40 - vy40 - vx21 + vy21 >= 1
 vx40 - vy40 - vx24 + vy24 >= 1
 vx42 - vy42 - vx21 + vy21 >= 1
 vx42 - vy42 - vx24 + vy24 >= 1
 vx40 - vy40 - vx24 + vy24 >= 1
 vx40 - vy40 - vx23 + vy23 >= 1
 vx42 - vy42 - vx24 + vy24 >= 1
 vx42 - vy42 - vx23 + vy23 >= 1
 vx40 - vy40 - vx23 + vy23 >= 1
 vx40 - vy40 - vx20 + vy20 >= 1
 vx42 - vy42 - vx23 + vy23 >= 1
 vx42 - vy42 - vx20 + vy20 >= 1
 vx40 - vy40 - vx20 + vy20 >= 1
 vx40 - vy40 - vx25 + vy25 >= 1
 vx42 - vy42 - vx20 + vy20 >= 1
 vx42 - vy42 - vx25 + vy25 >= 1
 vx40 - vy40 - vx25 + vy25 >= 1
 vx40 - vy40 - vx22 + vy22 >= 1
 vx42 - vy42 - vx25 + vy25 >= 1
 vx42 - vy42 - vx22 + vy22 >= 1
 vx40 - vy40 - vx41 + vy41 <= -1
 vx40 - vy40 - vx64 + vy64 <= -1
 vx42 - vy42 - vx41 + vy41 <= -1
 vx42 - vy42 - vx64 + vy64 <= -1
 vx40 - vy40 - vx43 + vy43 >= 1
 vx40 - vy40 - vx44 + vy44 >= 1
 vx42 - vy42 - vx43 + vy43 >= 1
 vx42 - vy42 - vx44 + vy44 >= 1
 vx40 - vy40 - vx44 + vy44 >= 1
 vx40 - vy40 - vx3 + vy3 >= 1
 vx42 - vy42 - vx44 + vy44 >= 1
 vx42 - vy42 - vx3 + vy3 >= 1
 vx40 - vy40 - vx3 + vy3 >= 1
 vx40 - vy40 - vx45 + vy45 >= 1
 vx42 - vy42 - vx3 + vy3 >= 1
 vx42 - vy42 - vx45 + vy45 >= 1
 vx40 - vy40 - vx45 + vy45 >= 1
 vx40 - vy40 - vx46 + vy46 >= 1
 vx42 - vy42 - vx45 + vy45 >= 1
 vx42 - vy42 - vx46 + vy46 >= 1
 vx40 - vy40 - vx46 + vy46 >= 1
 vx40 - vy40 - vx47 + vy47 >= 1
 vx42 - vy42 - vx46 + vy46 >= 1
 vx42 - vy42 - vx47 + vy47 >= 1
 vx40 - vy40 - vx47 + vy47 >= 1
 vx40 - vy40 - vx48 + vy48 >= 1
 vx42 - vy42 - vx47 + vy47 >= 1
 vx42 - vy42 - vx48 + vy48 >= 1
 vx40 - vy40 - vx48 + vy48 >= 1
 vx40 - vy40 - vx35 + vy35 >= 1
 vx42 - vy42 - vx48 + vy48 >= 1
 vx42 - vy42 - vx35 + vy35 >= 1
 vx40 - vy40 - vx35 + vy35 >= 1
 vx40 - vy40 - vx49 + vy49 >= 1
 vx42 - vy42 - vx35 + vy35 >= 1
 vx42 - vy42 - vx49 + vy49 >= 1
 vx40 - vy40 - vx49 + vy49 >= 1
 vx40 - vy40 - vx50 + vy50 >= 1
 vx42 - vy42 - vx49 + vy49 >= 1
 vx42 - vy42 - vx50 + vy50 >= 1
 vx40 - vy40 - vx50 + vy50 >= 1
 vx40 - vy40 - vx51 + vy51 >= 1
 vx42 - vy42 - vx50 + vy50 >= 1
 vx42 - vy42 - vx51 + vy51 >= 1
 vx40 - vy40 - vx51 + vy51 >= 1
 vx40 - vy40 - vx52 + vy52 >= 1
 vx42 - vy42 - vx51 + vy51 >= 1
 vx42 - vy42 - vx52 + vy52 >= 1
 vx40 - vy40 - vx52 + vy52 >= 1
 vx40 - vy40 - vx62 + vy62 >= 1
 vx42 - vy42 - vx52 + vy52 >= 1
 vx42 - vy42 - vx62 + vy62 >= 1
 vx40 - vy40 - vx62 + vy62 >= 1
 vx40 - vy40 - vx53 + vy53 >= 1
 vx42 - vy42 - vx62 + vy62 >= 1
 vx42 - vy42 - vx53 + vy53 >= 1
 vx40 - vy40 - vx53 + vy53 >= 1
 vx40 - vy40 - vx54 + vy54 >= 1
 vx42 - vy42 - vx53 + vy53 >= 1
 vx42 - vy42 - vx54 + vy54 >= 1
 vx40 - vy40 - vx54 + vy54 >= 1
 vx40 - vy40 - vx55 + vy55 >= 1
 vx42 - vy42 - vx54 + vy54 >= 1
 vx42 - vy42 - vx55 + vy55 >= 1
 vx40 - vy40 - vx55 + vy55 >= 1
 vx40 - vy40 - vx56 + vy56 >= 1
 vx42 - vy42 - vx55 + vy55 >= 1
 vx42 - vy42 - vx56 + vy56 >= 1
 vx40 - vy40 - vx56 + vy56 >= 1
 vx40 - vy40 - vx57 + vy57 >= 1
 vx42 - vy42 - vx56 + vy56 >= 1
 vx42 - vy42 - vx57 + vy57 >= 1
 vx40 - vy40 - vx57 + vy57 >= 1
 vx40 - vy40 - vx58 + vy58 >= 1
 vx42 - vy42 - vx57 + vy57 >= 1
 vx42 - vy42 - vx58 + vy58 >= 1
 vx40 - vy40 - vx58 + vy58 >= 1
 vx40 - vy40 - vx59 + vy59 >= 1
 vx42 - vy42 - vx58 + vy58 >= 1
 vx42 - vy42 - vx59 + vy59 >= 1
 vx40 - vy40 - vx59 + vy59 >= 1
 vx40 - vy40 - vx60 + vy60 >= 1
 vx42 - vy42 - vx59 + vy59 >= 1
 vx42 - vy42 - vx60 + vy60 >= 1
 vx40 - vy40 - vx60 + vy60 >= 1
 vx40 - vy40 - vx61 + vy61 >= 1
 vx42 - vy42 - vx60 + vy60 >= 1
 vx42 - vy42 - vx61 + vy61 >= 1
 q1 - oa1 - ob1 - oc1 - od1 = 0
 a1 + a2 - 2 ua1 - oa1 = 0
 b1 + b2 - 2 ub1 - ob1 = 0
 c1 + c2 - 2 uc1 - oc1 = 0
 d1 + d2 - 2 ud1 - od1 = 0
 vx42 - vy42 - vx63 + vy63 >= 1
 vx42 - vy42 - vx15 + vy15 >= 1
 vx5 - vy5 - vx63 + vy63 >= 1
 vx5 - vy5 - vx15 + vy15 >= 1
 vx42 - vy42 - vx15 + vy15 >= 1
 vx42 - vy42 - vx2 + vy2 >= 1
 vx5 - vy5 - vx15 + vy15 >= 1
 vx5 - vy5 - vx2 + vy2 >= 1
 vx42 - vy42 - vx2 + vy2 >= 1
 vx42 - vy42 - vx0 + vy0 >= 1
 vx5 - vy5 - vx2 + vy2 >= 1
 vx5 - vy5 - vx0 + vy0 >= 1
 vx42 - vy42 - vx0 + vy0 >= 1
 vx42 - vy42 - vx14 + vy14 >= 1
 vx5 - vy5 - vx0 + vy0 >= 1
 vx5 - vy5 - vx14 + vy14 >= 1
 vx42 - vy42 - vx14 + vy14 >= 1
 vx42 - vy42 - vx16 + vy16 >= 1
 vx5 - vy5 - vx14 + vy14 >= 1
 vx5 - vy5 - vx16 + vy16 >= 1
 vx42 - vy42 - vx16 + vy16 >= 1
 vx42 - vy42 - vx10 + vy10 >= 1
 vx5 - vy5 - vx16 + vy16 >= 1
 vx5 - vy5 - vx10 + vy10 >= 1
 vx42 - vy42 - vx10 + vy10 >= 1
 vx42 - vy42 - vx12 + vy12 >= 1
 vx5 - vy5 - vx10 + vy10 >= 1
 vx5 - vy5 - vx12 + vy12 >= 1
 vx42 - vy42 - vx12 + vy12 >= 1
 vx42 - vy42 - vx9 + vy9 >= 1
 vx5 - vy5 - vx12 + vy12 >= 1
 vx5 - vy5 - vx9 + vy9 >= 1
 vx42 - vy42 - vx9 + vy9 >= 1
 vx42 - vy42 - vx1 + vy1 >= 1
 vx5 - vy5 - vx9 + vy9 >= 1
 vx5 - vy5 - vx1 + vy1 >= 1
 vx42 - vy42 - vx1 + vy1 >= 1
 vx42 - vy42 - vx7 + vy7 >= 1
 vx5 - vy5 - vx1 + vy1 >= 1
 vx5 - vy5 - vx7 + vy7 >= 1
 vx42 - vy42 - vx7 + vy7 >= 1
 vx42 - vy42 - vx6 + vy6 >= 1
 vx5 - vy5 - vx7 + vy7 >= 1
 vx5 - vy5 - vx6 + vy6 >= 1
 vx42 - vy42 - vx6 + vy6 >= 1
 vx42 - vy42 - vx8 + vy8 >= 1
 vx5 - vy5 - vx6 + vy6 >= 1
 vx5 - vy5 - vx8 + vy8 >= 1
 vx42 - vy42 - vx8 + vy8 >= 1
 vx42 - vy42 - vx11 + vy11 >= 1
 vx5 - vy5 - vx8 + vy8 >= 1
 vx5 - vy5 - vx11 + vy11 >= 1
 vx42 - vy42 - vx11 + vy11 >= 1
 vx42 - vy42 - vx13 + vy13 >= 1
 vx5 - vy5 - vx11 + vy11 >= 1
 vx5 - vy5 - vx13 + vy13 >= 1
 vx42 - vy42 - vx13 + vy13 >= 1
 vx42 - vy42 - vx4 + vy4 >= 1
 vx5 - vy5 - vx13 + vy13 >= 1
 vx5 - vy5 - vx4 + vy4 >= 1
 vx42 - vy42 - vx4 + vy4 >= 1
 vx42 - vy42 - vx36 + vy36 >= 1
 vx5 - vy5 - vx4 + vy4 >= 1
 vx5 - vy5 - vx36 + vy36 >= 1
 vx42 - vy42 - vx36 + vy36 >= 1
 vx42 - vy42 - vx39 + vy39 >= 1
 vx5 - vy5 - vx36 + vy36 >= 1
 vx5 - vy5 - vx39 + vy39 >= 1
 vx42 - vy42 - vx39 + vy39 >= 1
 vx42 - vy42 - vx38 + vy38 >= 1
 vx5 - vy5 - vx39 + vy39 >= 1
 vx5 - vy5 - vx38 + vy38 >= 1
 vx42 - vy42 - vx38 + vy38 >= 1
 vx42 - vy42 - vx37 + vy37 >= 1
 vx5 - vy5 - vx38 + vy38 >= 1
 vx5 - vy5 - vx37 + vy37 >= 1
 vx42 + vy42 - vx17 - vy17 >= 1
 vx42 + vy42 - vx27 - vy27 >= 1
 vx5 + vy5 - vx17 - vy17 >= 1
 vx5 + vy5 - vx27 - vy27 >= 1
 vx42 + vy42 - vx27 - vy27 >= 1
 vx42 + vy42 - vx30 - vy30 >= 1
 vx5 + vy5 - vx27 - vy27 >= 1
 vx5 + vy5 - vx30 - vy30 >= 1
 vx42 + vy42 - vx30 - vy30 >= 1
 vx42 + vy42 - vx31 - vy31 >= 1
 vx5 + vy5 - vx30 - vy30 >= 1
 vx5 + vy5 - vx31 - vy31 >= 1
 vx42 + vy42 - vx31 - vy31 >= 1
 vx42 + vy42 - vx33 - vy33 >= 1
 vx5 + vy5 - vx31 - vy31 >= 1
 vx5 + vy5 - vx33 - vy33 >= 1
 vx42 + vy42 - vx33 - vy33 >= 1
 vx42 + vy42 - vx29 - vy29 >= 1
 vx5 + vy5 - vx33 - vy33 >= 1
 vx5 + vy5 - vx29 - vy29 >= 1
 vx42 + vy42 - vx29 - vy29 >= 1
 vx42 + vy42 - vx34 - vy34 >= 1
 vx5 + vy5 - vx29 - vy29 >= 1
 vx5 + vy5 - vx34 - vy34 >= 1
 vx42 - vx34 >= 1
 vx42 - vx32 >= 1
 vx5 - vx34 >= 1
 vx5 - vx32 >= 1
 vx42 - vy42 - vx32 + vy32 >= 1
 vx42 - vy42 - vx65 + vy65 >= 1
 vx5 - vy5 - vx32 + vy32 >= 1
 vx5 - vy5 - vx65 + vy65 >= 1
 vx42 - vy42 - vx65 + vy65 >= 1
 vx42 - vy42 - vx28 + vy28 >= 1
 vx5 - vy5 - vx65 + vy65 >= 1
 vx5 - vy5 - vx28 + vy28 >= 1
 vx42 - vy42 - vx28 + vy28 >= 1
 vx42 - vy42 - vx18 + vy18 >= 1
 vx5 - vy5 - vx28 + vy28 >= 1
 vx5 - vy5 - vx18 + vy18 >= 1
 vx42 - vy42 - vx18 + vy18 >= 1
 vx42 - vy42 - vx19 + vy19 >= 1
 vx5 - vy5 - vx18 + vy18 >= 1
 vx5 - vy5 - vx19 + vy19 >= 1
 vx42 - vy42 - vx19 + vy19 >= 1
 vx42 - vy42 - vx26 + vy26 >= 1
 vx5 - vy5 - vx19 + vy19 >= 1
 vx5 - vy5 - vx26 + vy26 >= 1
 vx42 - vy42 - vx26 + vy26 >= 1
 vx42 - vy42 - vx21 + vy21 >= 1
 vx5 - vy5 - vx26 + vy26 >= 1
 vx5 - vy5 - vx21 + vy21 >= 1
 vx42 - vy42 - vx21 + vy21 >= 1
 vx42 - vy42 - vx24 + vy24 >= 1
 vx5 - vy5 - vx21 + vy21 >= 1
 vx5 - vy5 - vx24 + vy24 >= 1
 vx42 - vy42 - vx24 + vy24 >= 1
 vx42 - vy42 - vx23 + vy23 >= 1
 vx5 - vy5 - vx24 + vy24 >= 1
 vx5 - vy5 - vx23 + vy23 >= 1
 vx42 - vy42 - vx23 + vy23 >= 1
 vx42 - vy42 - vx20 + vy20 >= 1
 vx5 - vy5 - vx23 + vy23 >= 1
 vx5 - vy5 - vx20 + vy20 >= 1
 vx42 - vy42 - vx20 + vy20 >= 1
 vx42 - vy42 - vx25 + vy25 >= 1
 vx5 - vy5 - vx20 + vy20 >= 1
 vx5 - vy5 - vx25 + vy25 >= 1
 vx42 - vy42 - vx25 + vy25 >= 1
 vx42 - vy42 - vx22 + vy22 >= 1
 vx5 - vy5 - vx25 + vy25 >= 1
 vx5 - vy5 - vx22 + vy22 >= 1
 vx42 + vy42 - vx41 - vy41 <= -1
 vx42 + vy42 - vx64 - vy64 <= -1
 vx5 + vy5 - vx41 - vy41 <= -1
 vx5 + vy5 - vx64 - vy64 <= -1
 vy42 - vy64 <= -1
 vy42 - vy43 <= -1
 vy5 - vy64 <= -1
 vy5 - vy43 <= -1
 vy42 - vy43 <= -1
 vy42 - vy44 <= -1
 vy5 - vy43 <= -1
 vy5 - vy44 <= -1
 vx42 - vy42 - vx44 + vy44 >= 1
 vx42 - vy42 - vx3 + vy3 >= 1
 vx5 - vy5 - vx44 + vy44 >= 1
 vx5 - vy5 - vx3 + vy3 >= 1
 vx42 - vy42 - vx3 + vy3 >= 1
 vx42 - vy42 - vx45 + vy45 >= 1
 vx5 - vy5 - vx3 + vy3 >= 1
 vx5 - vy5 - vx45 + vy45 >= 1
 vx42 - vy42 - vx45 + vy45 >= 1
 vx42 - vy42 - vx46 + vy46 >= 1
 vx5 - vy5 - vx45 + vy45 >= 1
 vx5 - vy5 - vx46 + vy46 >= 1
 vx42 - vy42 - vx46 + vy46 >= 1
 vx42 - vy42 - vx47 + vy47 >= 1
 vx5 - vy5 - vx46 + vy46 >= 1
 vx5 - vy5 - vx47 + vy47 >= 1
 vx42 - vy42 - vx47 + vy47 >= 1
 vx42 - vy42 - vx48 + vy48 >= 1
 vx5 - vy5 - vx47 + vy47 >= 1
 vx5 - vy5 - vx48 + vy48 >= 1
 vx42 - vy42 - vx48 + vy48 >= 1
 vx42 - vy42 - vx35 + vy35 >= 1
 vx5 - vy5 - vx48 + vy48 >= 1
 vx5 - vy5 - vx35 + vy35 >= 1
 vx42 - vy42 - vx35 + vy35 >= 1
 vx42 - vy42 - vx49 + vy49 >= 1
 vx5 - vy5 - vx35 + vy35 >= 1
 vx5 - vy5 - vx49 + vy49 >= 1
 vx42 - vy42 - vx49 + vy49 >= 1
 vx42 - vy42 - vx50 + vy50 >= 1
 vx5 - vy5 - vx49 + vy49 >= 1
 vx5 - vy5 - vx50 + vy50 >= 1
 vx42 - vy42 - vx50 + vy50 >= 1
 vx42 - vy42 - vx51 + vy51 >= 1
 vx5 - vy5 - vx50 + vy50 >= 1
 vx5 - vy5 - vx51 + vy51 >= 1
 vx42 - vy42 - vx51 + vy51 >= 1
 vx42 - vy42 - vx52 + vy52 >= 1
 vx5 - vy5 - vx51 + vy51 >= 1
 vx5 - vy5 - vx52 + vy52 >= 1
 vx42 - vy42 - vx52 + vy52 >= 1
 vx42 - vy42 - vx62 + vy62 >= 1
 vx5 - vy5 - vx52 + vy52 >= 1
 vx5 - vy5 - vx62 + vy62 >= 1
 vx42 - vy42 - vx62 + vy62 >= 1
 vx42 - vy42 - vx53 + vy53 >= 1
 vx5 - vy5 - vx62 + vy62 >= 1
 vx5 - vy5 - vx53 + vy53 >= 1
 vx42 - vy42 - vx53 + vy53 >= 1
 vx42 - vy42 - vx54 + vy54 >= 1
 vx5 - vy5 - vx53 + vy53 >= 1
 vx5 - vy5 - vx54 + vy54 >= 1
 vx42 - vy42 - vx54 + vy54 >= 1
 vx42 - vy42 - vx55 + vy55 >= 1
 vx5 - vy5 - vx54 + vy54 >= 1
 vx5 - vy5 - vx55 + vy55 >= 1
 vx42 - vy42 - vx55 + vy55 >= 1
 vx42 - vy42 - vx56 + vy56 >= 1
 vx5 - vy5 - vx55 + vy55 >= 1
 vx5 - vy5 - vx56 + vy56 >= 1
 vx42 - vy42 - vx56 + vy56 >= 1
 vx42 - vy42 - vx57 + vy57 >= 1
 vx5 - vy5 - vx56 + vy56 >= 1
 vx5 - vy5 - vx57 + vy57 >= 1
 vx42 - vy42 - vx57 + vy57 >= 1
 vx42 - vy42 - vx58 + vy58 >= 1
 vx5 - vy5 - vx57 + vy57 >= 1
 vx5 - vy5 - vx58 + vy58 >= 1
 vx42 - vy42 - vx58 + vy58 >= 1
 vx42 - vy42 - vx59 + vy59 >= 1
 vx5 - vy5 - vx58 + vy58 >= 1
 vx5 - vy5 - vx59 + vy59 >= 1
 vx42 - vy42 - vx59 + vy59 >= 1
 vx42 - vy42 - vx60 + vy60 >= 1
 vx5 - vy5 - vx59 + vy59 >= 1
 vx5 - vy5 - vx60 + vy60 >= 1
 vx42 - vy42 - vx60 + vy60 >= 1
 vx42 - vy42 - vx61 + vy61 >= 1
 vx5 - vy5 - vx60 + vy60 >= 1
 vx5 - vy5 - vx61 + vy61 >= 1
 q2 - oa2 - ob2 - oc2 - od2 = 0
 a2 + a3 - 2 ua2 - oa2 = 0
 b2 + b3 - 2 ub2 - ob2 = 0
 c2 + c3 - 2 uc2 - oc2 = 0
 d2 + d3 - 2 ud2 - od2 = 0
 vx5 - vy5 - vx15 + vy15 >= 1
 vx5 - vy5 - vx2 + vy2 >= 1
 vx63 - vy63 - vx15 + vy15 >= 1
 vx63 - vy63 - vx2 + vy2 >= 1
 vx5 - vy5 - vx2 + vy2 >= 1
 vx5 - vy5 - vx0 + vy0 >= 1
 vx63 - vy63 - vx2 + vy2 >= 1
 vx63 - vy63 - vx0 + vy0 >= 1
 vx5 - vy5 - vx0 + vy0 >= 1
 vx5 - vy5 - vx14 + vy14 >= 1
 vx63 - vy63 - vx0 + vy0 >= 1
 vx63 - vy63 - vx14 + vy14 >= 1
 vx5 - vy5 - vx14 + vy14 >= 1
 vx5 - vy5 - vx16 + vy16 >= 1
 vx63 - vy63 - vx14 + vy14 >= 1
 vx63 - vy63 - vx16 + vy16 >= 1
 vx5 - vy5 - vx16 + vy16 >= 1
 vx5 - vy5 - vx10 + vy10 >= 1
 vx63 - vy63 - vx16 + vy16 >= 1
 vx63 - vy63 - vx10 + vy10 >= 1
 vx5 - vy5 - vx10 + vy10 >= 1
 vx5 - vy5 - vx12 + vy12 >= 1
 vx63 - vy63 - vx10 + vy10 >= 1
 vx63 - vy63 - vx12 + vy12 >= 1
 vx5 - vy5 - vx12 + vy12 >= 1
 vx5 - vy5 - vx9 + vy9 >= 1
 vx63 - vy63 - vx12 + vy12 >= 1
 vx63 - vy63 - vx9 + vy9 >= 1
 vx5 - vy5 - vx9 + vy9 >= 1
 vx5 - vy5 - vx1 + vy1 >= 1
 vx63 - vy63 - vx9 + vy9 >= 1
 vx63 - vy63 - vx1 + vy1 >= 1
 vx5 - vy5 - vx1 + vy1 >= 1
 vx5 - vy5 - vx7 + vy7 >= 1
 vx63 - vy63 - vx1 + vy1 >= 1
 vx63 - vy63 - vx7 + vy7 >= 1
 vx5 - vy5 - vx7 + vy7 >= 1
 vx5 - vy5 - vx6 + vy6 >= 1
 vx63 - vy63 - vx7 + vy7 >= 1
 vx63 - vy63 - vx6 + vy6 >= 1
 vx5 - vy5 - vx6 + vy6 >= 1
 vx5 - vy5 - vx8 + vy8 >= 1
 vx63 - vy63 - vx6 + vy6 >= 1
 vx63 - vy63 - vx8 + vy8 >= 1
 vx5 - vy5 - vx8 + vy8 >= 1
 vx5 - vy5 - vx11 + vy11 >= 1
 vx63 - vy63 - vx8 + vy8 >= 1
 vx63 - vy63 - vx11 + vy11 >= 1
 vx5 - vy5 - vx11 + vy11 >= 1
 vx5 - vy5 - vx13 + vy13 >= 1
 vx63 - vy63 - vx11 + vy11 >= 1
 vx63 - vy63 - vx13 + vy13 >= 1
 vx5 - vy5 - vx13 + vy13 >= 1
 vx5 - vy5 - vx4 + vy4 >= 1
 vx63 - vy63 - vx13 + vy13 >= 1
 vx63 - vy63 - vx4 + vy4 >= 1
 vx5 - vy5 - vx4 + vy4 >= 1
 vx5 - vy5 - vx36 + vy36 >= 1
 vx63 - vy63 - vx4 + vy4 >= 1
 vx63 - vy63 - vx36 + vy36 >= 1
 vx5 - vy5 - vx36 + vy36 >= 1
 vx5 - vy5 - vx39 + vy39 >= 1
 vx63 - vy63 - vx36 + vy36 >= 1
 vx63 - vy63 - vx39 + vy39 >= 1
 vx5 - vy5 - vx39 + vy39 >= 1
 vx5 - vy5 - vx38 + vy38 >= 1
 vx63 - vy63 - vx39 + vy39 >= 1
 vx63 - vy63 - vx38 + vy38 >= 1
 vx5 - vy5 - vx38 + vy38 >= 1
 vx5 - vy5 - vx37 + vy37 >= 1
 vx63 - vy63 - vx38 + vy38 >= 1
 vx63 - vy63 - vx37 + vy37 >= 1
 vx5 + vy5 - vx17 - vy17 >= 1
 vx5 + vy5 - vx27 - vy27 >= 1
 vx63 + vy63 - vx17 - vy17 >= 1
 vx63 + vy63 - vx27 - vy27 >= 1
 vx5 + vy5 - vx27 - vy27 >= 1
 vx5 + vy5 - vx30 - vy30 >= 1
 vx63 + vy63 - vx27 - vy27 >= 1
 vx63 + vy63 - vx30 - vy30 >= 1
 vx5 + vy5 - vx30 - vy30 >= 1
 vx5 + vy5 - vx31 - vy31 >= 1
 vx63 + vy63 - vx30 - vy30 >= 1
 vx63 + vy63 - vx31 - vy31 >= 1
 vx5 + vy5 - vx31 - vy31 >= 1
 vx5 + vy5 - vx33 - vy33 >= 1
 vx63 + vy63 - vx31 - vy31 >= 1
 vx63 + vy63 - vx33 - vy33 >= 1
 vx5 + vy5 - vx33 - vy33 >= 1
 vx5 + vy5 - vx29 - vy29 >= 1
 vx63 + vy63 - vx33 - vy33 >= 1
 vx63 + vy63 - vx29 - vy29 >= 1
 vx5 + vy5 - vx29 - vy29 >= 1
 vx5 + vy5 - vx34 - vy34 >= 1
 vx63 + vy63 - vx29 - vy29 >= 1
 vx63 + vy63 - vx34 - vy34 >= 1
 vx5 + vy5 - vx34 - vy34 >= 1
 vx5 + vy5 - vx32 - vy32 >= 1
 vx63 + vy63 - vx34 - vy34 >= 1
 vx63 + vy63 - vx32 - vy32 >= 1
 vx5 - vx32 >= 1
 vx5 - vx65 >= 1
 vx63 - vx32 >= 1
 vx63 - vx65 >= 1
 vx5 - vy5 - vx65 + vy65 >= 1
 vx5 - vy5 - vx28 + vy28 >= 1
 vx63 - vy63 - vx65 + vy65 >= 1
 vx63 - vy63 - vx28 + vy28 >= 1
 vx5 - vy5 - vx28 + vy28 >= 1
 vx5 - vy5 - vx18 + vy18 >= 1
 vx63 - vy63 - vx28 + vy28 >= 1
 vx63 - vy63 - vx18 + vy18 >= 1
 vx5 - vy5 - vx18 + vy18 >= 1
 vx5 - vy5 - vx19 + vy19 >= 1
 vx63 - vy63 - vx18 + vy18 >= 1
 vx63 - vy63 - vx19 + vy19 >= 1
 vx5 - vy5 - vx19 + vy19 >= 1
 vx5 - vy5 - vx26 + vy26 >= 1
 vx63 - vy63 - vx19 + vy19 >= 1
 vx63 - vy63 - vx26 + vy26 >= 1
 vx5 - vy5 - vx26 + vy26 >= 1
 vx5 - vy5 - vx21 + vy21 >= 1
 vx63 - vy63 - vx26 + vy26 >= 1
 vx63 - vy63 - vx21 + vy21 >= 1
 vx5 - vy5 - vx21 + vy21 >= 1
 vx5 - vy5 - vx24 + vy24 >= 1
 vx63 - vy63 - vx21 + vy21 >= 1
 vx63 - vy63 - vx24 + vy24 >= 1
 vx5 - vy5 - vx24 + vy24 >= 1
 vx5 - vy5 - vx23 + vy23 >= 1
 vx63 - vy63 - vx24 + vy24 >= 1
 vx63 - vy63 - vx23 + vy23 >= 1
 vx5 - vy5 - vx23 + vy23 >= 1
 vx5 - vy5 - vx20 + vy20 >= 1
 vx63 - vy63 - vx23 + vy23 >= 1
 vx63 - vy63 - vx20 + vy20 >= 1
 vx5 - vy5 - vx20 + vy20 >= 1
 vx5 - vy5 - vx25 + vy25 >= 1
 vx63 - vy63 - vx20 + vy20 >= 1
 vx63 - vy63 - vx25 + vy25 >= 1
 vx5 - vy5 - vx25 + vy25 >= 1
 vx5 - vy5 - vx22 + vy22 >= 1
 vx63 - vy63 - vx25 + vy25 >= 1
 vx63 - vy63 - vx22 + vy22 >= 1
 vx5 + vy5 - vx41 - vy41 <= -1
 vx5 + vy5 - vx64 - vy64 <= -1
 vx63 + vy63 - vx41 - vy41 <= -1
 vx63 + vy63 - vx64 - vy64 <= -1
 vx5 + vy5 - vx64 - vy64 <= -1
 vx5 + vy5 - vx43 - vy43 <= -1
 vx63 + vy63 - vx64 - vy64 <= -1
 vx63 + vy63 - vx43 - vy43 <= -1
 vx5 + vy5 - vx43 - vy43 <= -1
 vx5 + vy5 - vx44 - vy44 <= -1
 vx63 + vy63 - vx43 - vy43 <= -1
 vx63 + vy63 - vx44 - vy44 <= -1
 vy5 - vy44 <= -1
 vy5 - vy3 <= -1
 vy63 - vy44 <= -1
 vy63 - vy3 <= -1
 vx5 + vy5 - vx45 - vy45 >= 1
 vx5 + vy5 - vx46 - vy46 >= 1
 vx63 + vy63 - vx45 - vy45 >= 1
 vx63 + vy63 - vx46 - vy46 >= 1
 vx5 + vy5 - vx46 - vy46 >= 1
 vx5 + vy5 - vx47 - vy47 >= 1
 vx63 + vy63 - vx46 - vy46 >= 1
 vx63 + vy63 - vx47 - vy47 >= 1
 vx5 - vy5 - vx47 + vy47 >= 1
 vx5 - vy5 - vx48 + vy48 >= 1
 vx63 - vy63 - vx47 + vy47 >= 1
 vx63 - vy63 - vx48 + vy48 >= 1
 vx5 - vy5 - vx48 + vy48 >= 1
 vx5 - vy5 - vx35 + vy35 >= 1
 vx63 - vy63 - vx48 + vy48 >= 1
 vx63 - vy63 - vx35 + vy35 >= 1
 vx5 - vy5 - vx35 + vy35 >= 1
 vx5 - vy5 - vx49 + vy49 >= 1
 vx63 - vy63 - vx35 + vy35 >= 1
 vx63 - vy63 - vx49 + vy49 >= 1
 vx5 - vy5 - vx49 + vy49 >= 1
 vx5 - vy5 - vx50 + vy50 >= 1
 vx63 - vy63 - vx49 + vy49 >= 1
 vx63 - vy63 - vx50 + vy50 >= 1
 vx5 - vy5 - vx50 + vy50 >= 1
 vx5 - vy5 - vx51 + vy51 >= 1
 vx63 - vy63 - vx50 + vy50 >= 1
 vx63 - vy63 - vx51 + vy51 >= 1
 vx5 - vy5 - vx51 + vy51 >= 1
 vx5 - vy5 - vx52 + vy52 >= 1
 vx63 - vy63 - vx51 + vy51 >= 1
 vx63 - vy63 - vx52 + vy52 >= 1
 vx5 - vy5 - vx52 + vy52 >= 1
 vx5 - vy5 - vx62 + vy62 >= 1
 vx63 - vy63 - vx52 + vy52 >= 1
 vx63 - vy63 - vx62 + vy62 >= 1
 vx5 - vy5 - vx62 + vy62 >= 1
 vx5 - vy5 - vx53 + vy53 >= 1
 vx63 - vy63 - vx62 + vy62 >= 1
 vx63 - vy63 - vx53 + vy53 >= 1
 vx5 - vy5 - vx53 + vy53 >= 1
 vx5 - vy5 - vx54 + vy54 >= 1
 vx63 - vy63 - vx53 + vy53 >= 1
 vx63 - vy63 - vx54 + vy54 >= 1
 vx5 - vy5 - vx54 + vy54 >= 1
 vx5 - vy5 - vx55 + vy55 >= 1
 vx63 - vy63 - vx54 + vy54 >= 1
 vx63 - vy63 - vx55 + vy55 >= 1
 vx5 - vy5 - vx55 + vy55 >= 1
 vx5 - vy5 - vx56 + vy56 >= 1
 vx63 - vy63 - vx55 + vy55 >= 1
 vx63 - vy63 - vx56 + vy56 >= 1
 vx5 - vy5 - vx56 + vy56 >= 1
 vx5 - vy5 - vx57 + vy57 >= 1
 vx63 - vy63 - vx56 + vy56 >= 1
 vx63 - vy63 - vx57 + vy57 >= 1
 vx5 - vy5 - vx57 + vy57 >= 1
 vx5 - vy5 - vx58 + vy58 >= 1
 vx63 - vy63 - vx57 + vy57 >= 1
 vx63 - vy63 - vx58 + vy58 >= 1
 vx5 - vy5 - vx58 + vy58 >= 1
 vx5 - vy5 - vx59 + vy59 >= 1
 vx63 - vy63 - vx58 + vy58 >= 1
 vx63 - vy63 - vx59 + vy59 >= 1
 vx5 - vy5 - vx59 + vy59 >= 1
 vx5 - vy5 - vx60 + vy60 >= 1
 vx63 - vy63 - vx59 + vy59 >= 1
 vx63 - vy63 - vx60 + vy60 >= 1
 vx5 - vy5 - vx60 + vy60 >= 1
 vx5 - vy5 - vx61 + vy61 >= 1
 vx63 - vy63 - vx60 + vy60 >= 1
 vx63 - vy63 - vx61 + vy61 >= 1
 q3 - oa3 - ob3 - oc3 - od3 = 0
 a3 + a4 - 2 ua3 - oa3 = 0
 b3 + b4 - 2 ub3 - ob3 = 0
 c3 + c4 - 2 uc3 - oc3 = 0
 d3 + d4 - 2 ud3 - od3 = 0
 vx63 - vy63 - vx2 + vy2 >= 1
 vx63 - vy63 - vx0 + vy0 >= 1
 vx15 - vy15 - vx2 + vy2 >= 1
 vx15 - vy15 - vx0 + vy0 >= 1
 vx63 - vy63 - vx0 + vy0 >= 1
 vx63 - vy63 - vx14 + vy14 >= 1
 vx15 - vy15 - vx0 + vy0 >= 1
 vx15 - vy15 - vx14 + vy14 >= 1
 vx63 - vy63 - vx14 + vy14 >= 1
 vx63 - vy63 - vx16 + vy16 >= 1
 vx15 - vy15 - vx14 + vy14 >= 1
 vx15 - vy15 - vx16 + vy16 >= 1
 vx63 - vy63 - vx16 + vy16 >= 1
 vx63 - vy63 - vx10 + vy10 >= 1
 vx15 - vy15 - vx16 + vy16 >= 1
 vx15 - vy15 - vx10 + vy10 >= 1
 vx63 - vy63 - vx10 + vy10 >= 1
 vx63 - vy63 - vx12 + vy12 >= 1
 vx15 - vy15 - vx10 + vy10 >= 1
 vx15 - vy15 - vx12 + vy12 >= 1
 vx63 - vy63 - vx12 + vy12 >= 1
 vx63 - vy63 - vx9 + vy9 >= 1
 vx15 - vy15 - vx12 + vy12 >= 1
 vx15 - vy15 - vx9 + vy9 >= 1
 vx63 - vy63 - vx9 + vy9 >= 1
 vx63 - vy63 - vx1 + vy1 >= 1
 vx15 - vy15 - vx9 + vy9 >= 1
 vx15 - vy15 - vx1 + vy1 >= 1
 vx63 - vy63 - vx1 + vy1 >= 1
 vx63 - vy63 - vx7 + vy7 >= 1
 vx15 - vy15 - vx1 + vy1 >= 1
 vx15 - vy15 - vx7 + vy7 >= 1
 vx63 - vy63 - vx7 + vy7 >= 1
 vx63 - vy63 - vx6 + vy6 >= 1
 vx15 - vy15 - vx7 + vy7 >= 1
 vx15 - vy15 - vx6 + vy6 >= 1
 vx63 - vy63 - vx6 + vy6 >= 1
 vx63 - vy63 - vx8 + vy8 >= 1
 vx15 - vy15 - vx6 + vy6 >= 1
 vx15 - vy15 - vx8 + vy8 >= 1
 vx63 - vy63 - vx8 + vy8 >= 1
 vx63 - vy63 - vx11 + vy11 >= 1
 vx15 - vy15 - vx8 + vy8 >= 1
 vx15 - vy15 - vx11 + vy11 >= 1
 vx63 - vy63 - vx11 + vy11 >= 1
 vx63 - vy63 - vx13 + vy13 >= 1
 vx15 - vy15 - vx11 + vy11 >= 1
 vx15 - vy15 - vx13 + vy13 >= 1
 vx63 - vy63 - vx13 + vy13 >= 1
 vx63 - vy63 - vx4 + vy4 >= 1
 vx15 - vy15 - vx13 + vy13 >= 1
 vx15 - vy15 - vx4 + vy4 >= 1
 vx63 - vy63 - vx4 + vy4 >= 1
 vx63 - vy63 - vx36 + vy36 >= 1
 vx15 - vy15 - vx4 + vy4 >= 1
 vx15 - vy15 - vx36 + vy36 >= 1
 vx63 - vy63 - vx36 + vy36 >= 1
 vx63 - vy63 - vx39 + vy39 >= 1
 vx15 - vy15 - vx36 + vy36 >= 1
 vx15 - vy15 - vx39 + vy39 >= 1
 vx63 - vy63 - vx39 + vy39 >= 1
 vx63 - vy63 - vx38 + vy38 >= 1
 vx15 - vy15 - vx39 + vy39 >= 1
 vx15 - vy15 - vx38 + vy38 >= 1
 vx63 - vy63 - vx38 + vy38 >= 1
 vx63 - vy63 - vx37 + vy37 >= 1
 vx15 - vy15 - vx38 + vy38 >= 1
 vx15 - vy15 - vx37 + vy37 >= 1
 vx63 + vy63 - vx17 - vy17 >= 1
 vx63 + vy63 - vx27 - vy27 >= 1
 vx15 + vy15 - vx17 - vy17 >= 1
 vx15 + vy15 - vx27 - vy27 >= 1
 vx63 + vy63 - vx27 - vy27 >= 1
 vx63 + vy63 - vx30 - vy30 >= 1
 vx15 + vy15 - vx27 - vy27 >= 1
 vx15 + vy15 - vx30 - vy30 >= 1
 vx63 + vy63 - vx30 - vy30 >= 1
 vx63 + vy63 - vx31 - vy31 >= 1
 vx15 + vy15 - vx30 - vy30 >= 1
 vx15 + vy15 - vx31 - vy31 >= 1
 vx63 + vy63 - vx31 - vy31 >= 1
 vx63 + vy63 - vx33 - vy33 >= 1
 vx15 + vy15 - vx31 - vy31 >= 1
 vx15 + vy15 - vx33 - vy33 >= 1
 vx63 + vy63 - vx33 - vy33 >= 1
 vx63 + vy63 - vx29 - vy29 >= 1
 vx15 + vy15 - vx33 - vy33 >= 1
 vx15 + vy15 - vx29 - vy29 >= 1
 vx63 + vy63 - vx29 - vy29 >= 1
 vx63 + vy63 - vx34 - vy34 >= 1
 vx15 + vy15 - vx29 - vy29 >= 1
 vx15 + vy15 - vx34 - vy34 >= 1
 vx63 + vy63 - vx34 - vy34 >= 1
 vx63 + vy63 - vx32 - vy32 >= 1
 vx15 + vy15 - vx34 - vy34 >= 1
 vx15 + vy15 - vx32 - vy32 >= 1
 vx63 - vx32 >= 1
 vx63 - vx65 >= 1
 vx15 - vx32 >= 1
 vx15 - vx65 >= 1
 vx63 - vx65 >= 1
 vx63 - vx28 >= 1
 vx15 - vx65 >= 1
 vx15 - vx28 >= 1
 vx63 - vx28 >= 1
 vx63 - vx18 >= 1
 vx15 - vx28 >= 1
 vx15 - vx18 >= 1
 vx63 - vy63 - vx18 + vy18 >= 1
 vx63 - vy63 - vx19 + vy19 >= 1
 vx15 - vy15 - vx18 + vy18 >= 1
 vx15 - vy15 - vx19 + vy19 >= 1
 vx63 - vy63 - vx19 + vy19 >= 1
 vx63 - vy63 - vx26 + vy26 >= 1
 vx15 - vy15 - vx19 + vy19 >= 1
 vx15 - vy15 - vx26 + vy26 >= 1
 vx63 - vy63 - vx26 + vy26 >= 1
 vx63 - vy63 - vx21 + vy21 >= 1
 vx15 - vy15 - vx26 + vy26 >= 1
 vx15 - vy15 - vx21 + vy21 >= 1
 vx63 - vy63 - vx21 + vy21 >= 1
 vx63 - vy63 - vx24 + vy24 >= 1
 vx15 - vy15 - vx21 + vy21 >= 1
 vx15 - vy15 - vx24 + vy24 >= 1
 vx63 - vy63 - vx24 + vy24 >= 1
 vx63 - vy63 - vx23 + vy23 >= 1
 vx15 - vy15 - vx24 + vy24 >= 1
 vx15 - vy15 - vx23 + vy23 >= 1
 vx63 - vy63 - vx23 + vy23 >= 1
 vx63 - vy63 - vx20 + vy20 >= 1
 vx15 - vy15 - vx23 + vy23 >= 1
 vx15 - vy15 - vx20 + vy20 >= 1
 vx63 - vy63 - vx20 + vy20 >= 1
 vx63 - vy63 - vx25 + vy25 >= 1
 vx15 - vy15 - vx20 + vy20 >= 1
 vx15 - vy15 - vx25 + vy25 >= 1
 vx63 - vy63 - vx25 + vy25 >= 1
 vx63 - vy63 - vx22 + vy22 >= 1
 vx15 - vy15 - vx25 + vy25 >= 1
 vx15 - vy15 - vx22 + vy22 >= 1
 vx63 - vy63 - vx41 + vy41 <= -1
 vx63 - vy63 - vx64 + vy64 <= -1
 vx15 - vy15 - vx41 + vy41 <= -1
 vx15 - vy15 - vx64 + vy64 <= -1
 vx63 - vy63 - vx64 + vy64 <= -1
 vx63 - vy63 - vx43 + vy43 <= -1
 vx15 - vy15 - vx64 + vy64 <= -1
 vx15 - vy15 - vx43 + vy43 <= -1
 vx63 - vx43 <= -1
 vx63 - vx44 <= -1
 vx15 - vx43 <= -1
 vx15 - vx44 <= -1
 vx63 - vx44 <= -1
 vx63 - vx3 <= -1
 vx15 - vx44 <= -1
 vx15 - vx3 <= -1
 vx63 + vy63 - vx45 - vy45 >= 1
 vx63 + vy63 - vx46 - vy46 >= 1
 vx15 + vy15 - vx45 - vy45 >= 1
 vx15 + vy15 - vx46 - vy46 >= 1
 vx63 + vy63 - vx46 - vy46 >= 1
 vx63 + vy63 - vx47 - vy47 >= 1
 vx15 + vy15 - vx46 - vy46 >= 1
 vx15 + vy15 - vx47 - vy47 >= 1
 vx63 - vx47 >= 1
 vx63 - vx48 >= 1
 vx15 - vx47 >= 1
 vx15 - vx48 >= 1
 vx63 - vx48 >= 1
 vx63 - vx35 >= 1
 vx15 - vx48 >= 1
 vx15 - vx35 >= 1
 vx63 - vx35 >= 1
 vx63 - vx49 >= 1
 vx15 - vx35 >= 1
 vx15 - vx49 >= 1
 vx63 - vx49 >= 1
 vx63 - vx50 >= 1
 vx15 - vx49 >= 1
 vx15 - vx50 >= 1
 vx63 - vx50 >= 1
 vx63 - vx51 >= 1
 vx15 - vx50 >= 1
 vx15 - vx51 >= 1
 vx63 - vy63 - vx51 + vy51 >= 1
 vx63 - vy63 - vx52 + vy52 >= 1
 vx15 - vy15 - vx51 + vy51 >= 1
 vx15 - vy15 - vx52 + vy52 >= 1
 vx63 - vy63 - vx52 + vy52 >= 1
 vx63 - vy63 - vx62 + vy62 >= 1
 vx15 - vy15 - vx52 + vy52 >= 1
 vx15 - vy15 - vx62 + vy62 >= 1
 vx63 - vy63 - vx62 + vy62 >= 1
 vx63 - vy63 - vx53 + vy53 >= 1
 vx15 - vy15 - vx62 + vy62 >= 1
 vx15 - vy15 - vx53 + vy53 >= 1
 vx63 - vy63 - vx53 + vy53 >= 1
 vx63 - vy63 - vx54 + vy54 >= 1
 vx15 - vy15 - vx53 + vy53 >= 1
 vx15 - vy15 - vx54 + vy54 >= 1
 vx63 - vy63 - vx54 + vy54 >= 1
 vx63 - vy63 - vx55 + vy55 >= 1
 vx15 - vy15 - vx54 + vy54 >= 1
 vx15 - vy15 - vx55 + vy55 >= 1
 vx63 - vy63 - vx55 + vy55 >= 1
 vx63 - vy63 - vx56 + vy56 >= 1
 vx15 - vy15 - vx55 + vy55 >= 1
 vx15 - vy15 - vx56 + vy56 >= 1
 vx63 - vy63 - vx56 + vy56 >= 1
 vx63 - vy63 - vx57 + vy57 >= 1
 vx15 - vy15 - vx56 + vy56 >= 1
 vx15 - vy15 - vx57 + vy57 >= 1
 vx63 - vy63 - vx57 + vy57 >= 1
 vx63 - vy63 - vx58 + vy58 >= 1
 vx15 - vy15 - vx57 + vy57 >= 1
 vx15 - vy15 - vx58 + vy58 >= 1
 vx63 - vy63 - vx58 + vy58 >= 1
 vx63 - vy63 - vx59 + vy59 >= 1
 vx15 - vy15 - vx58 + vy58 >= 1
 vx15 - vy15 - vx59 + vy59 >= 1
 vx63 - vy63 - vx59 + vy59 >= 1
 vx63 - vy63 - vx60 + vy60 >= 1
 vx15 - vy15 - vx59 + vy59 >= 1
 vx15 - vy15 - vx60 + vy60 >= 1
 vx63 - vy63 - vx60 + vy60 >= 1
 vx63 - vy63 - vx61 + vy61 >= 1
 vx15 - vy15 - vx60 + vy60 >= 1
 vx15 - vy15 - vx61 + vy61 >= 1
 q4 - oa4 - ob4 - oc4 - od4 = 0
 a4 + a5 - 2 ua4 - oa4 = 0
 b4 + b5 - 2 ub4 - ob4 = 0
 c4 + c5 - 2 uc4 - oc4 = 0
 d4 + d5 - 2 ud4 - od4 = 0
 vx15 - vy15 - vx0 + vy0 >= 1
 vx15 - vy15 - vx14 + vy14 >= 1
 vx2 - vy2 - vx0 + vy0 >= 1
 vx2 - vy2 - vx14 + vy14 >= 1
 vx15 - vy15 - vx14 + vy14 >= 1
 vx15 - vy15 - vx16 + vy16 >= 1
 vx2 - vy2 - vx14 + vy14 >= 1
 vx2 - vy2 - vx16 + vy16 >= 1
 vx15 - vy15 - vx16 + vy16 >= 1
 vx15 - vy15 - vx10 + vy10 >= 1
 vx2 - vy2 - vx16 + vy16 >= 1
 vx2 - vy2 - vx10 + vy10 >= 1
 vx15 - vy15 - vx10 + vy10 >= 1
 vx15 - vy15 - vx12 + vy12 >= 1
 vx2 - vy2 - vx10 + vy10 >= 1
 vx2 - vy2 - vx12 + vy12 >= 1
 vx15 - vy15 - vx12 + vy12 >= 1
 vx15 - vy15 - vx9 + vy9 >= 1
 vx2 - vy2 - vx12 + vy12 >= 1
 vx2 - vy2 - vx9 + vy9 >= 1
 vx15 - vy15 - vx9 + vy9 >= 1
 vx15 - vy15 - vx1 + vy1 >= 1
 vx2 - vy2 - vx9 + vy9 >= 1
 vx2 - vy2 - vx1 + vy1 >= 1
 vx15 - vy15 - vx1 + vy1 >= 1
 vx15 - vy15 - vx7 + vy7 >= 1
 vx2 - vy2 - vx1 + vy1 >= 1
 vx2 - vy2 - vx7 + vy7 >= 1
 vx15 - vy15 - vx7 + vy7 >= 1
 vx15 - vy15 - vx6 + vy6 >= 1
 vx2 - vy2 - vx7 + vy7 >= 1
 vx2 - vy2 - vx6 + vy6 >= 1
 vx15 - vy15 - vx6 + vy6 >= 1
 vx15 - vy15 - vx8 + vy8 >= 1
 vx2 - vy2 - vx6 + vy6 >= 1
 vx2 - vy2 - vx8 + vy8 >= 1
 vx15 - vy15 - vx8 + vy8 >= 1
 vx15 - vy15 - vx11 + vy11 >= 1
 vx2 - vy2 - vx8 + vy8 >= 1
 vx2 - vy2 - vx11 + vy11 >= 1
 vx15 - vy15 - vx11 + vy11 >= 1
 vx15 - vy15 - vx13 + vy13 >= 1
 vx2 - vy2 - vx11 + vy11 >= 1
 vx2 - vy2 - vx13 + vy13 >= 1
 vx15 - vy15 - vx13 + vy13 >= 1
 vx15 - vy15 - vx4 + vy4 >= 1
 vx2 - vy2 - vx13 + vy13 >= 1
 vx2 - vy2 - vx4 + vy4 >= 1
 vx15 - vy15 - vx4 + vy4 >= 1
 vx15 - vy15 - vx36 + vy36 >= 1
 vx2 - vy2 - vx4 + vy4 >= 1
 vx2 - vy2 - vx36 + vy36 >= 1
 vx15 - vy15 - vx36 + vy36 >= 1
 vx15 - vy15 - vx39 + vy39 >= 1
 vx2 - vy2 - vx36 + vy36 >= 1
 vx2 - vy2 - vx39 + vy39 >= 1
 vx15 - vy15 - vx39 + vy39 >= 1
 vx15 - vy15 - vx38 + vy38 >= 1
 vx2 - vy2 - vx39 + vy39 >= 1
 vx2 - vy2 - vx38 + vy38 >= 1
 vx15 - vy15 - vx38 + vy38 >= 1
 vx15 - vy15 - vx37 + vy37 >= 1
 vx2 - vy2 - vx38 + vy38 >= 1
 vx2 - vy2 - vx37 + vy37 >= 1
 vx15 + vy15 - vx17 - vy17 >= 1
 vx15 + vy15 - vx27 - vy27 >= 1
 vx2 + vy2 - vx17 - vy17 >= 1
 vx2 + vy2 - vx27 - vy27 >= 1
 vx15 + vy15 - vx27 - vy27 >= 1
 vx15 + vy15 - vx30 - vy30 >= 1
 vx2 + vy2 - vx27 - vy27 >= 1
 vx2 + vy2 - vx30 - vy30 >= 1
 vx15 + vy15 - vx30 - vy30 >= 1
 vx15 + vy15 - vx31 - vy31 >= 1
 vx2 + vy2 - vx30 - vy30 >= 1
 vx2 + vy2 - vx31 - vy31 >= 1
 vx15 + vy15 - vx31 - vy31 >= 1
 vx15 + vy15 - vx33 - vy33 >= 1
 vx2 + vy2 - vx31 - vy31 >= 1
 vx2 + vy2 - vx33 - vy33 >= 1
 vx15 + vy15 - vx33 - vy33 >= 1
 vx15 + vy15 - vx29 - vy29 >= 1
 vx2 + vy2 - vx33 - vy33 >= 1
 vx2 + vy2 - vx29 - vy29 >= 1
 vx15 + vy15 - vx29 - vy29 >= 1
 vx15 + vy15 - vx34 - vy34 >= 1
 vx2 + vy2 - vx29 - vy29 >= 1
 vx2 + vy2 - vx34 - vy34 >= 1
 vx15 + vy15 - vx34 - vy34 >= 1
 vx15 + vy15 - vx32 - vy32 >= 1
 vx2 + vy2 - vx34 - vy34 >= 1
 vx2 + vy2 - vx32 - vy32 >= 1
 vx15 + vy15 - vx32 - vy32 >= 1
 vx15 + vy15 - vx65 - vy65 >= 1
 vx2 + vy2 - vx32 - vy32 >= 1
 vx2 + vy2 - vx65 - vy65 >= 1
 vx15 + vy15 - vx65 - vy65 >= 1
 vx15 + vy15 - vx28 - vy28 >= 1
 vx2 + vy2 - vx65 - vy65 >= 1
 vx2 + vy2 - vx28 - vy28 >= 1
 vx15 + vy15 - vx28 - vy28 >= 1
 vx15 + vy15 - vx18 - vy18 >= 1
 vx2 + vy2 - vx28 - vy28 >= 1
 vx2 + vy2 - vx18 - vy18 >= 1
 vx15 - vx18 >= 1
 vx15 - vx19 >= 1
 vx2 - vx18 >= 1
 vx2 - vx19 >= 1
 vx15 - vx19 >= 1
 vx15 - vx26 >= 1
 vx2 - vx19 >= 1
 vx2 - vx26 >= 1
 vx15 - vy15 - vx26 + vy26 >= 1
 vx15 - vy15 - vx21 + vy21 >= 1
 vx2 - vy2 - vx26 + vy26 >= 1
 vx2 - vy2 - vx21 + vy21 >= 1
 vx15 - vy15 - vx21 + vy21 >= 1
 vx15 - vy15 - vx24 + vy24 >= 1
 vx2 - vy2 - vx21 + vy21 >= 1
 vx2 - vy2 - vx24 + vy24 >= 1
 vx15 - vy15 - vx24 + vy24 >= 1
 vx15 - vy15 - vx23 + vy23 >= 1
 vx2 - vy2 - vx24 + vy24 >= 1
 vx2 - vy2 - vx23 + vy23 >= 1
 vx15 - vy15 - vx23 + vy23 >= 1
 vx15 - vy15 - vx20 + vy20 >= 1
 vx2 - vy2 - vx23 + vy23 >= 1
 vx2 - vy2 - vx20 + vy20 >= 1
 vx15 - vy15 - vx20 + vy20 >= 1
 vx15 - vy15 - vx25 + vy25 >= 1
 vx2 - vy2 - vx20 + vy20 >= 1
 vx2 - vy2 - vx25 + vy25 >= 1
 vx15 - vy15 - vx25 + vy25 >= 1
 vx15 - vy15 - vx22 + vy22 >= 1
 vx2 - vy2 - vx25 + vy25 >= 1
 vx2 - vy2 - vx22 + vy22 >= 1
 vx15 - vy15 - vx41 + vy41 <= -1
 vx15 - vy15 - vx64 + vy64 <= -1
 vx2 - vy2 - vx41 + vy41 <= -1
 vx2 - vy2 - vx64 + vy64 <= -1
 vx15 - vy15 - vx64 + vy64 <= -1
 vx15 - vy15 - vx43 + vy43 <= -1
 vx2 - vy2 - vx64 + vy64 <= -1
 vx2 - vy2 - vx43 + vy43 <= -1
 vx15 - vy15 - vx43 + vy43 <= -1
 vx15 - vy15 - vx44 + vy44 <= -1
 vx2 - vy2 - vx43 + vy43 <= -1
 vx2 - vy2 - vx44 + vy44 <= -1
 vx15 - vy15 - vx44 + vy44 <= -1
 vx15 - vy15 - vx3 + vy3 <= -1
 vx2 - vy2 - vx44 + vy44 <= -1
 vx2 - vy2 - vx3 + vy3 <= -1
 vy15 - vy3 >= 1
 vy15 - vy45 >= 1
 vy2 - vy3 >= 1
 vy2 - vy45 >= 1
 vx15 + vy15 - vx45 - vy45 >= 1
 vx15 + vy15 - vx46 - vy46 >= 1
 vx2 + vy2 - vx45 - vy45 >= 1
 vx2 + vy2 - vx46 - vy46 >= 1
 vx15 + vy15 - vx46 - vy46 >= 1
 vx15 + vy15 - vx47 - vy47 >= 1
 vx2 + vy2 - vx46 - vy46 >= 1
 vx2 + vy2 - vx47 - vy47 >= 1
 vx15 + vy15 - vx47 - vy47 >= 1
 vx15 + vy15 - vx48 - vy48 >= 1
 vx2 + vy2 - vx47 - vy47 >= 1
 vx2 + vy2 - vx48 - vy48 >= 1
 vx15 + vy15 - vx48 - vy48 >= 1
 vx15 + vy15 - vx35 - vy35 >= 1
 vx2 + vy2 - vx48 - vy48 >= 1
 vx2 + vy2 - vx35 - vy35 >= 1
 vx15 + vy15 - vx35 - vy35 >= 1
 vx15 + vy15 - vx49 - vy49 >= 1
 vx2 + vy2 - vx35 - vy35 >= 1
 vx2 + vy2 - vx49 - vy49 >= 1
 vx15 + vy15 - vx49 - vy49 >= 1
 vx15 + vy15 - vx50 - vy50 >= 1
 vx2 + vy2 - vx49 - vy49 >= 1
 vx2 + vy2 - vx50 - vy50 >= 1
 vx15 - vx50 >= 1
 vx15 - vx51 >= 1
 vx2 - vx50 >= 1
 vx2 - vx51 >= 1
 vx15 - vx51 >= 1
 vx15 - vx52 >= 1
 vx2 - vx51 >= 1
 vx2 - vx52 >= 1
 vx15 - vy15 - vx52 + vy52 >= 1
 vx15 - vy15 - vx62 + vy62 >= 1
 vx2 - vy2 - vx52 + vy52 >= 1
 vx2 - vy2 - vx62 + vy62 >= 1
 vx15 - vy15 - vx62 + vy62 >= 1
 vx15 - vy15 - vx53 + vy53 >= 1
 vx2 - vy2 - vx62 + vy62 >= 1
 vx2 - vy2 - vx53 + vy53 >= 1
 vx15 - vy15 - vx53 + vy53 >= 1
 vx15 - vy15 - vx54 + vy54 >= 1
 vx2 - vy2 - vx53 + vy53 >= 1
 vx2 - vy2 - vx54 + vy54 >= 1
 vx15 - vy15 - vx54 + vy54 >= 1
 vx15 - vy15 - vx55 + vy55 >= 1
 vx2 - vy2 - vx54 + vy54 >= 1
 vx2 - vy2 - vx55 + vy55 >= 1
 vx15 - vy15 - vx55 + vy55 >= 1
 vx15 - vy15 - vx56 + vy56 >= 1
 vx2 - vy2 - vx55 + vy55 >= 1
 vx2 - vy2 - vx56 + vy56 >= 1
 vx15 - vy15 - vx56 + vy56 >= 1
 vx15 - vy15 - vx57 + vy57 >= 1
 vx2 - vy2 - vx56 + vy56 >= 1
 vx2 - vy2 - vx57 + vy57 >= 1
 vx15 - vy15 - vx57 + vy57 >= 1
 vx15 - vy15 - vx58 + vy58 >= 1
 vx2 - vy2 - vx57 + vy57 >= 1
 vx2 - vy2 - vx58 + vy58 >= 1
 vx15 - vy15 - vx58 + vy58 >= 1
 vx15 - vy15 - vx59 + vy59 >= 1
 vx2 - vy2 - vx58 + vy58 >= 1
 vx2 - vy2 - vx59 + vy59 >= 1
 vx15 - vy15 - vx59 + vy59 >= 1
 vx15 - vy15 - vx60 + vy60 >= 1
 vx2 - vy2 - vx59 + vy59 >= 1
 vx2 - vy2 - vx60 + vy60 >= 1
 vx15 - vy15 - vx60 + vy60 >= 1
 vx15 - vy15 - vx61 + vy61 >= 1
 vx2 - vy2 - vx60 + vy60 >= 1
 vx2 - vy2 - vx61 + vy61 >= 1
 q5 - oa5 - ob5 - oc5 - od5 = 0
 a5 + a6 - 2 ua5 - oa5 = 0
 b5 + b6 - 2 ub5 - ob5 = 0
 c5 + c6 - 2 uc5 - oc5 = 0
 d5 + d6 - 2 ud5 - od5 = 0
 vx2 - vy2 - vx14 + vy14 >= 1
 vx2 - vy2 - vx16 + vy16 >= 1
 vx0 - vy0 - vx14 + vy14 >= 1
 vx0 - vy0 - vx16 + vy16 >= 1
 vx2 - vy2 - vx16 + vy16 >= 1
 vx2 - vy2 - vx10 + vy10 >= 1
 vx0 - vy0 - vx16 + vy16 >= 1
 vx0 - vy0 - vx10 + vy10 >= 1
 vx2 - vy2 - vx10 + vy10 >= 1
 vx2 - vy2 - vx12 + vy12 >= 1
 vx0 - vy0 - vx10 + vy10 >= 1
 vx0 - vy0 - vx12 + vy12 >= 1
 vx2 - vy2 - vx12 + vy12 >= 1
 vx2 - vy2 - vx9 + vy9 >= 1
 vx0 - vy0 - vx12 + vy12 >= 1
 vx0 - vy0 - vx9 + vy9 >= 1
 vx2 - vy2 - vx9 + vy9 >= 1
 vx2 - vy2 - vx1 + vy1 >= 1
 vx0 - vy0 - vx9 + vy9 >= 1
 vx0 - vy0 - vx1 + vy1 >= 1
 vx2 - vy2 - vx1 + vy1 >= 1
 vx2 - vy2 - vx7 + vy7 >= 1
 vx0 - vy0 - vx1 + vy1 >= 1
 vx0 - vy0 - vx7 + vy7 >= 1
 vx2 - vy2 - vx7 + vy7 >= 1
 vx2 - vy2 - vx6 + vy6 >= 1
 vx0 - vy0 - vx7 + vy7 >= 1
 vx0 - vy0 - vx6 + vy6 >= 1
 vx2 - vy2 - vx6 + vy6 >= 1
 vx2 - vy2 - vx8 + vy8 >= 1
 vx0 - vy0 - vx6 + vy6 >= 1
 vx0 - vy0 - vx8 + vy8 >= 1
 vx2 - vy2 - vx8 + vy8 >= 1
 vx2 - vy2 - vx11 + vy11 >= 1
 vx0 - vy0 - vx8 + vy8 >= 1
 vx0 - vy0 - vx11 + vy11 >= 1
 vx2 - vy2 - vx11 + vy11 >= 1
 vx2 - vy2 - vx13 + vy13 >= 1
 vx0 - vy0 - vx11 + vy11 >= 1
 vx0 - vy0 - vx13 + vy13 >= 1
 vx2 - vy2 - vx13 + vy13 >= 1
 vx2 - vy2 - vx4 + vy4 >= 1
 vx0 - vy0 - vx13 + vy13 >= 1
 vx0 - vy0 - vx4 + vy4 >= 1
 vx2 - vy2 - vx4 + vy4 >= 1
 vx2 - vy2 - vx36 + vy36 >= 1
 vx0 - vy0 - vx4 + vy4 >= 1
 vx0 - vy0 - vx36 + vy36 >= 1
 vx2 - vy2 - vx36 + vy36 >= 1
 vx2 - vy2 - vx39 + vy39 >= 1
 vx0 - vy0 - vx36 + vy36 >= 1
 vx0 - vy0 - vx39 + vy39 >= 1
 vx2 - vy2 - vx39 + vy39 >= 1
 vx2 - vy2 - vx38 + vy38 >= 1
 vx0 - vy0 - vx39 + vy39 >= 1
 vx0 - vy0 - vx38 + vy38 >= 1
 vx2 + vy2 - vx38 - vy38 >= 1
 vx2 + vy2 - vx37 - vy37 >= 1
 vx0 + vy0 - vx38 - vy38 >= 1
 vx0 + vy0 - vx37 - vy37 >= 1
 vx2 + vy2 - vx17 - vy17 >= 1
 vx2 + vy2 - vx27 - vy27 >= 1
 vx0 + vy0 - vx17 - vy17 >= 1
 vx0 + vy0 - vx27 - vy27 >= 1
 vx2 + vy2 - vx27 - vy27 >= 1
 vx2 + vy2 - vx30 - vy30 >= 1
 vx0 + vy0 - vx27 - vy27 >= 1
 vx0 + vy0 - vx30 - vy30 >= 1
 vx2 + vy2 - vx30 - vy30 >= 1
 vx2 + vy2 - vx31 - vy31 >= 1
 vx0 + vy0 - vx30 - vy30 >= 1
 vx0 + vy0 - vx31 - vy31 >= 1
 vx2 + vy2 - vx31 - vy31 >= 1
 vx2 + vy2 - vx33 - vy33 >= 1
 vx0 + vy0 - vx31 - vy31 >= 1
 vx0 + vy0 - vx33 - vy33 >= 1
 vx2 + vy2 - vx33 - vy33 >= 1
 vx2 + vy2 - vx29 - vy29 >= 1
 vx0 + vy0 - vx33 - vy33 >= 1
 vx0 + vy0 - vx29 - vy29 >= 1
 vx2 + vy2 - vx29 - vy29 >= 1
 vx2 + vy2 - vx34 - vy34 >= 1
 vx0 + vy0 - vx29 - vy29 >= 1
 vx0 + vy0 - vx34 - vy34 >= 1
 vx2 + vy2 - vx34 - vy34 >= 1
 vx2 + vy2 - vx32 - vy32 >= 1
 vx0 + vy0 - vx34 - vy34 >= 1
 vx0 + vy0 - vx32 - vy32 >= 1
 vx2 + vy2 - vx32 - vy32 >= 1
 vx2 + vy2 - vx65 - vy65 >= 1
 vx0 + vy0 - vx32 - vy32 >= 1
 vx0 + vy0 - vx65 - vy65 >= 1
 vx2 + vy2 - vx65 - vy65 >= 1
 vx2 + vy2 - vx28 - vy28 >= 1
 vx0 + vy0 - vx65 - vy65 >= 1
 vx0 + vy0 - vx28 - vy28 >= 1
 vx2 + vy2 - vx28 - vy28 >= 1
 vx2 + vy2 - vx18 - vy18 >= 1
 vx0 + vy0 - vx28 - vy28 >= 1
 vx0 + vy0 - vx18 - vy18 >= 1
 vx2 + vy2 - vx18 - vy18 >= 1
 vx2 + vy2 - vx19 - vy19 >= 1
 vx0 + vy0 - vx18 - vy18 >= 1
 vx0 + vy0 - vx19 - vy19 >= 1
 vx2 + vy2 - vx19 - vy19 >= 1
 vx2 + vy2 - vx26 - vy26 >= 1
 vx0 + vy0 - vx19 - vy19 >= 1
 vx0 + vy0 - vx26 - vy26 >= 1
 vx2 - vx26 >= 1
 vx2 - vx21 >= 1
 vx0 - vx26 >= 1
 vx0 - vx21 >= 1
 vx2 - vy2 - vx21 + vy21 >= 1
 vx2 - vy2 - vx24 + vy24 >= 1
 vx0 - vy0 - vx21 + vy21 >= 1
 vx0 - vy0 - vx24 + vy24 >= 1
 vx2 - vy2 - vx24 + vy24 >= 1
 vx2 - vy2 - vx23 + vy23 >= 1
 vx0 - vy0 - vx24 + vy24 >= 1
 vx0 - vy0 - vx23 + vy23 >= 1
 vx2 - vy2 - vx23 + vy23 >= 1
 vx2 - vy2 - vx20 + vy20 >= 1
 vx0 - vy0 - vx23 + vy23 >= 1
 vx0 - vy0 - vx20 + vy20 >= 1
 vx2 - vy2 - vx20 + vy20 >= 1
 vx2 - vy2 - vx25 + vy25 >= 1
 vx0 - vy0 - vx20 + vy20 >= 1
 vx0 - vy0 - vx25 + vy25 >= 1
 vx2 - vy2 - vx25 + vy25 >= 1
 vx2 - vy2 - vx22 + vy22 >= 1
 vx0 - vy0 - vx25 + vy25 >= 1
 vx0 - vy0 - vx22 + vy22 >= 1
 vx2 - vy2 - vx41 + vy41 <= -1
 vx2 - vy2 - vx64 + vy64 <= -1
 vx0 - vy0 - vx41 + vy41 <= -1
 vx0 - vy0 - vx64 + vy64 <= -1
 vx2 - vy2 - vx64 + vy64 <= -1
 vx2 - vy2 - vx43 + vy43 <= -1
 vx0 - vy0 - vx64 + vy64 <= -1
 vx0 - vy0 - vx43 + vy43 <= -1
 vx2 - vy2 - vx43 + vy43 <= -1
 vx2 - vy2 - vx44 + vy44 <= -1
 vx0 - vy0 - vx43 + vy43 <= -1
 vx0 - vy0 - vx44 + vy44 <= -1
 vx2 - vy2 - vx44 + vy44 <= -1
 vx2 - vy2 - vx3 + vy3 <= -1
 vx0 - vy0 - vx44 + vy44 <= -1
 vx0 - vy0 - vx3 + vy3 <= -1
 vy2 - vy3 >= 1
 vy2 - vy45 >= 1
 vy0 - vy3 >= 1
 vy0 - vy45 >= 1
 vx2 + vy2 - vx45 - vy45 >= 1
 vx2 + vy2 - vx46 - vy46 >= 1
 vx0 + vy0 - vx45 - vy45 >= 1
 vx0 + vy0 - vx46 - vy46 >= 1
 vx2 + vy2 - vx46 - vy46 >= 1
 vx2 + vy2 - vx47 - vy47 >= 1
 vx0 + vy0 - vx46 - vy46 >= 1
 vx0 + vy0 - vx47 - vy47 >= 1
 vx2 + vy2 - vx47 - vy47 >= 1
 vx2 + vy2 - vx48 - vy48 >= 1
 vx0 + vy0 - vx47 - vy47 >= 1
 vx0 + vy0 - vx48 - vy48 >= 1
 vx2 + vy2 - vx48 - vy48 >= 1
 vx2 + vy2 - vx35 - vy35 >= 1
 vx0 + vy0 - vx48 - vy48 >= 1
 vx0 + vy0 - vx35 - vy35 >= 1
 vx2 + vy2 - vx35 - vy35 >= 1
 vx2 + vy2 - vx49 - vy49 >= 1
 vx0 + vy0 - vx35 - vy35 >= 1
 vx0 + vy0 - vx49 - vy49 >= 1
 vx2 + vy2 - vx49 - vy49 >= 1
 vx2 + vy2 - vx50 - vy50 >= 1
 vx0 + vy0 - vx49 - vy49 >= 1
 vx0 + vy0 - vx50 - vy50 >= 1
 vx2 + vy2 - vx50 - vy50 >= 1
 vx2 + vy2 - vx51 - vy51 >= 1
 vx0 + vy0 - vx50 - vy50 >= 1
 vx0 + vy0 - vx51 - vy51 >= 1
 vx2 + vy2 - vx51 - vy51 >= 1
 vx2 + vy2 - vx52 - vy52 >= 1
 vx0 + vy0 - vx51 - vy51 >= 1
 vx0 + vy0 - vx52 - vy52 >= 1
 vx2 - vx52 >= 1
 vx2 - vx62 >= 1
 vx0 - vx52 >= 1
 vx0 - vx62 >= 1
 vx2 - vy2 - vx62 + vy62 >= 1
 vx2 - vy2 - vx53 + vy53 >= 1
 vx0 - vy0 - vx62 + vy62 >= 1
 vx0 - vy0 - vx53 + vy53 >= 1
 vx2 - vy2 - vx53 + vy53 >= 1
 vx2 - vy2 - vx54 + vy54 >= 1
 vx0 - vy0 - vx53 + vy53 >= 1
 vx0 - vy0 - vx54 + vy54 >= 1
 vx2 - vy2 - vx54 + vy54 >= 1
 vx2 - vy2 - vx55 + vy55 >= 1
 vx0 - vy0 - vx54 + vy54 >= 1
 vx0 - vy0 - vx55 + vy55 >= 1
 vx2 - vy2 - vx55 + vy55 >= 1
 vx2 - vy2 - vx56 + vy56 >= 1
 vx0 - vy0 - vx55 + vy55 >= 1
 vx0 - vy0 - vx56 + vy56 >= 1
 vx2 - vy2 - vx56 + vy56 >= 1
 vx2 - vy2 - vx57 + vy57 >= 1
 vx0 - vy0 - vx56 + vy56 >= 1
 vx0 - vy0 - vx57 + vy57 >= 1
 vx2 - vy2 - vx57 + vy57 >= 1
 vx2 - vy2 - vx58 + vy58 >= 1
 vx0 - vy0 - vx57 + vy57 >= 1
 vx0 - vy0 - vx58 + vy58 >= 1
 vx2 - vy2 - vx58 + vy58 >= 1
 vx2 - vy2 - vx59 + vy59 >= 1
 vx0 - vy0 - vx58 + vy58 >= 1
 vx0 - vy0 - vx59 + vy59 >= 1
 vx2 - vy2 - vx59 + vy59 >= 1
 vx2 - vy2 - vx60 + vy60 >= 1
 vx0 - vy0 - vx59 + vy59 >= 1
 vx0 - vy0 - vx60 + vy60 >= 1
 vx2 - vy2 - vx60 + vy60 >= 1
 vx2 - vy2 - vx61 + vy61 >= 1
 vx0 - vy0 - vx60 + vy60 >= 1
 vx0 - vy0 - vx61 + vy61 >= 1
 q6 - oa6 - ob6 - oc6 - od6 = 0
 a6 + a7 - 2 ua6 - oa6 = 0
 b6 + b7 - 2 ub6 - ob6 = 0
 c6 + c7 - 2 uc6 - oc6 = 0
 d6 + d7 - 2 ud6 - od6 = 0
 vx0 - vy0 - vx16 + vy16 >= 1
 vx0 - vy0 - vx10 + vy10 >= 1
 vx14 - vy14 - vx16 + vy16 >= 1
 vx14 - vy14 - vx10 + vy10 >= 1
 vx0 - vy0 - vx10 + vy10 >= 1
 vx0 - vy0 - vx12 + vy12 >= 1
 vx14 - vy14 - vx10 + vy10 >= 1
 vx14 - vy14 - vx12 + vy12 >= 1
 vx0 - vy0 - vx12 + vy12 >= 1
 vx0 - vy0 - vx9 + vy9 >= 1
 vx14 - vy14 - vx12 + vy12 >= 1
 vx14 - vy14 - vx9 + vy9 >= 1
 vx0 - vy0 - vx9 + vy9 >= 1
 vx0 - vy0 - vx1 + vy1 >= 1
 vx14 - vy14 - vx9 + vy9 >= 1
 vx14 - vy14 - vx1 + vy1 >= 1
 vx0 - vy0 - vx1 + vy1 >= 1
 vx0 - vy0 - vx7 + vy7 >= 1
 vx14 - vy14 - vx1 + vy1 >= 1
 vx14 - vy14 - vx7 + vy7 >= 1
 vx0 - vy0 - vx7 + vy7 >= 1
 vx0 - vy0 - vx6 + vy6 >= 1
 vx14 - vy14 - vx7 + vy7 >= 1
 vx14 - vy14 - vx6 + vy6 >= 1
 vx0 - vy0 - vx6 + vy6 >= 1
 vx0 - vy0 - vx8 + vy8 >= 1
 vx14 - vy14 - vx6 + vy6 >= 1
 vx14 - vy14 - vx8 + vy8 >= 1
 vx0 - vy0 - vx8 + vy8 >= 1
 vx0 - vy0 - vx11 + vy11 >= 1
 vx14 - vy14 - vx8 + vy8 >= 1
 vx14 - vy14 - vx11 + vy11 >= 1
 vx0 - vy0 - vx11 + vy11 >= 1
 vx0 - vy0 - vx13 + vy13 >= 1
 vx14 - vy14 - vx11 + vy11 >= 1
 vx14 - vy14 - vx13 + vy13 >= 1
 vx0 - vy0 - vx13 + vy13 >= 1
 vx0 - vy0 - vx4 + vy4 >= 1
 vx14 - vy14 - vx13 + vy13 >= 1
 vx14 - vy14 - vx4 + vy4 >= 1
 vx0 - vy0 - vx4 + vy4 >= 1
 vx0 - vy0 - vx36 + vy36 >= 1
 vx14 - vy14 - vx4 + vy4 >= 1
 vx14 - vy14 - vx36 + vy36 >= 1
 vx0 - vy0 - vx36 + vy36 >= 1
 vx0 - vy0 - vx39 + vy39 >= 1
 vx14 - vy14 - vx36 + vy36 >= 1
 vx14 - vy14 - vx39 + vy39 >= 1
 vx0 - vy0 - vx39 + vy39 >= 1
 vx0 - vy0 - vx38 + vy38 >= 1
 vx14 - vy14 - vx39 + vy39 >= 1
 vx14 - vy14 - vx38 + vy38 >= 1
 vx0 + vy0 - vx38 - vy38 >= 1
 vx0 + vy0 - vx37 - vy37 >= 1
 vx14 + vy14 - vx38 - vy38 >= 1
 vx14 + vy14 - vx37 - vy37 >= 1
 vx0 + vy0 - vx17 - vy17 >= 1
 vx0 + vy0 - vx27 - vy27 >= 1
 vx14 + vy14 - vx17 - vy17 >= 1
 vx14 + vy14 - vx27 - vy27 >= 1
 vx0 + vy0 - vx27 - vy27 >= 1
 vx0 + vy0 - vx30 - vy30 >= 1
 vx14 + vy14 - vx27 - vy27 >= 1
 vx14 + vy14 - vx30 - vy30 >= 1
 vx0 + vy0 - vx30 - vy30 >= 1
 vx0 + vy0 - vx31 - vy31 >= 1
 vx14 + vy14 - vx30 - vy30 >= 1
 vx14 + vy14 - vx31 - vy31 >= 1
 vx0 + vy0 - vx31 - vy31 >= 1
 vx0 + vy0 - vx33 - vy33 >= 1
 vx14 + vy14 - vx31 - vy31 >= 1
 vx14 + vy14 - vx33 - vy33 >= 1
 vx0 + vy0 - vx33 - vy33 >= 1
 vx0 + vy0 - vx29 - vy29 >= 1
 vx14 + vy14 - vx33 - vy33 >= 1
 vx14 + vy14 - vx29 - vy29 >= 1
 vx0 + vy0 - vx29 - vy29 >= 1
 vx0 + vy0 - vx34 - vy34 >= 1
 vx14 + vy14 - vx29 - vy29 >= 1
 vx14 + vy14 - vx34 - vy34 >= 1
 vx0 + vy0 - vx34 - vy34 >= 1
 vx0 + vy0 - vx32 - vy32 >= 1
 vx14 + vy14 - vx34 - vy34 >= 1
 vx14 + vy14 - vx32 - vy32 >= 1
 vx0 + vy0 - vx32 - vy32 >= 1
 vx0 + vy0 - vx65 - vy65 >= 1
 vx14 + vy14 - vx32 - vy32 >= 1
 vx14 + vy14 - vx65 - vy65 >= 1
 vx0 + vy0 - vx65 - vy65 >= 1
 vx0 + vy0 - vx28 - vy28 >= 1
 vx14 + vy14 - vx65 - vy65 >= 1
 vx14 + vy14 - vx28 - vy28 >= 1
 vx0 + vy0 - vx28 - vy28 >= 1
 vx0 + vy0 - vx18 - vy18 >= 1
 vx14 + vy14 - vx28 - vy28 >= 1
 vx14 + vy14 - vx18 - vy18 >= 1
 vx0 + vy0 - vx18 - vy18 >= 1
 vx0 + vy0 - vx19 - vy19 >= 1
 vx14 + vy14 - vx18 - vy18 >= 1
 vx14 + vy14 - vx19 - vy19 >= 1
 vx0 + vy0 - vx19 - vy19 >= 1
 vx0 + vy0 - vx26 - vy26 >= 1
 vx14 + vy14 - vx19 - vy19 >= 1
 vx14 + vy14 - vx26 - vy26 >= 1
 vx0 - vx26 >= 1
 vx0 - vx21 >= 1
 vx14 - vx26 >= 1
 vx14 - vx21 >= 1
 vx0 - vy0 - vx21 + vy21 >= 1
 vx0 - vy0 - vx24 + vy24 >= 1
 vx14 - vy14 - vx21 + vy21 >= 1
 vx14 - vy14 - vx24 + vy24 >= 1
 vx0 - vy0 - vx24 + vy24 >= 1
 vx0 - vy0 - vx23 + vy23 >= 1
 vx14 - vy14 - vx24 + vy24 >= 1
 vx14 - vy14 - vx23 + vy23 >= 1
 vx0 - vy0 - vx23 + vy23 >= 1
 vx0 - vy0 - vx20 + vy20 >= 1
 vx14 - vy14 - vx23 + vy23 >= 1
 vx14 - vy14 - vx20 + vy20 >= 1
 vx0 - vy0 - vx20 + vy20 >= 1
 vx0 - vy0 - vx25 + vy25 >= 1
 vx14 - vy14 - vx20 + vy20 >= 1
 vx14 - vy14 - vx25 + vy25 >= 1
 vx0 - vy0 - vx25 + vy25 >= 1
 vx0 - vy0 - vx22 + vy22 >= 1
 vx14 - vy14 - vx25 + vy25 >= 1
 vx14 - vy14 - vx22 + vy22 >= 1
 vx0 - vy0 - vx41 + vy41 <= -1
 vx0 - vy0 - vx64 + vy64 <= -1
 vx14 - vy14 - vx41 + vy41 <= -1
 vx14 - vy14 - vx64 + vy64 <= -1
 vx0 - vy0 - vx64 + vy64 <= -1
 vx0 - vy0 - vx43 + vy43 <= -1
 vx14 - vy14 - vx64 + vy64 <= -1
 vx14 - vy14 - vx43 + vy43 <= -1
 vx0 - vy0 - vx43 + vy43 <= -1
 vx0 - vy0 - vx44 + vy44 <= -1
 vx14 - vy14 - vx43 + vy43 <= -1
 vx14 - vy14 - vx44 + vy44 <= -1
 vx0 - vy0 - vx44 + vy44 <= -1
 vx0 - vy0 - vx3 + vy3 <= -1
 vx14 - vy14 - vx44 + vy44 <= -1
 vx14 - vy14 - vx3 + vy3 <= -1
 vy0 - vy3 >= 1
 vy0 - vy45 >= 1
 vy14 - vy3 >= 1
 vy14 - vy45 >= 1
 vy0 - vy45 >= 1
 vy0 - vy46 >= 1
 vy14 - vy45 >= 1
 vy14 - vy46 >= 1
 vx0 + vy0 - vx46 - vy46 >= 1
 vx0 + vy0 - vx47 - vy47 >= 1
 vx14 + vy14 - vx46 - vy46 >= 1
 vx14 + vy14 - vx47 - vy47 >= 1
 vx0 + vy0 - vx47 - vy47 >= 1
 vx0 + vy0 - vx48 - vy48 >= 1
 vx14 + vy14 - vx47 - vy47 >= 1
 vx14 + vy14 - vx48 - vy48 >= 1
 vx0 + vy0 - vx48 - vy48 >= 1
 vx0 + vy0 - vx35 - vy35 >= 1
 vx14 + vy14 - vx48 - vy48 >= 1
 vx14 + vy14 - vx35 - vy35 >= 1
 vx0 + vy0 - vx35 - vy35 >= 1
 vx0 + vy0 - vx49 - vy49 >= 1
 vx14 + vy14 - vx35 - vy35 >= 1
 vx14 + vy14 - vx49 - vy49 >= 1
 vx0 + vy0 - vx49 - vy49 >= 1
 vx0 + vy0 - vx50 - vy50 >= 1
 vx14 + vy14 - vx49 - vy49 >= 1
 vx14 + vy14 - vx50 - vy50 >= 1
 vx0 + vy0 - vx50 - vy50 >= 1
 vx0 + vy0 - vx51 - vy51 >= 1
 vx14 + vy14 - vx50 - vy50 >= 1
 vx14 + vy14 - vx51 - vy51 >= 1
 vx0 + vy0 - vx51 - vy51 >= 1
 vx0 + vy0 - vx52 - vy52 >= 1
 vx14 + vy14 - vx51 - vy51 >= 1
 vx14 + vy14 - vx52 - vy52 >= 1
 vx0 - vx52 >= 1
 vx0 - vx62 >= 1
 vx14 - vx52 >= 1
 vx14 - vx62 >= 1
 vx0 - vy0 - vx62 + vy62 >= 1
 vx0 - vy0 - vx53 + vy53 >= 1
 vx14 - vy14 - vx62 + vy62 >= 1
 vx14 - vy14 - vx53 + vy53 >= 1
 vx0 - vy0 - vx53 + vy53 >= 1
 vx0 - vy0 - vx54 + vy54 >= 1
 vx14 - vy14 - vx53 + vy53 >= 1
 vx14 - vy14 - vx54 + vy54 >= 1
 vx0 - vy0 - vx54 + vy54 >= 1
 vx0 - vy0 - vx55 + vy55 >= 1
 vx14 - vy14 - vx54 + vy54 >= 1
 vx14 - vy14 - vx55 + vy55 >= 1
 vx0 - vy0 - vx55 + vy55 >= 1
 vx0 - vy0 - vx56 + vy56 >= 1
 vx14 - vy14 - vx55 + vy55 >= 1
 vx14 - vy14 - vx56 + vy56 >= 1
 vx0 - vy0 - vx56 + vy56 >= 1
 vx0 - vy0 - vx57 + vy57 >= 1
 vx14 - vy14 - vx56 + vy56 >= 1
 vx14 - vy14 - vx57 + vy57 >= 1
 vx0 - vy0 - vx57 + vy57 >= 1
 vx0 - vy0 - vx58 + vy58 >= 1
 vx14 - vy14 - vx57 + vy57 >= 1
 vx14 - vy14 - vx58 + vy58 >= 1
 vx0 - vy0 - vx58 + vy58 >= 1
 vx0 - vy0 - vx59 + vy59 >= 1
 vx14 - vy14 - vx58 + vy58 >= 1
 vx14 - vy14 - vx59 + vy59 >= 1
 vx0 - vy0 - vx59 + vy59 >= 1
 vx0 - vy0 - vx60 + vy60 >= 1
 vx14 - vy14 - vx59 + vy59 >= 1
 vx14 - vy14 - vx60 + vy60 >= 1
 vx0 - vy0 - vx60 + vy60 >= 1
 vx0 - vy0 - vx61 + vy61 >= 1
 vx14 - vy14 - vx60 + vy60 >= 1
 vx14 - vy14 - vx61 + vy61 >= 1
 q7 - oa7 - ob7 - oc7 - od7 = 0
 a7 + a8 - 2 ua7 - oa7 = 0
 b7 + b8 - 2 ub7 - ob7 = 0
 c7 + c8 - 2 uc7 - oc7 = 0
 d7 + d8 - 2 ud7 - od7 = 0
 vx14 - vy14 - vx10 + vy10 >= 1
 vx14 - vy14 - vx12 + vy12 >= 1
 vx16 - vy16 - vx10 + vy10 >= 1
 vx16 - vy16 - vx12 + vy12 >= 1
 vx14 - vy14 - vx12 + vy12 >= 1
 vx14 - vy14 - vx9 + vy9 >= 1
 vx16 - vy16 - vx12 + vy12 >= 1
 vx16 - vy16 - vx9 + vy9 >= 1
 vx14 - vy14 - vx9 + vy9 >= 1
 vx14 - vy14 - vx1 + vy1 >= 1
 vx16 - vy16 - vx9 + vy9 >= 1
 vx16 - vy16 - vx1 + vy1 >= 1
 vx14 - vy14 - vx1 + vy1 >= 1
 vx14 - vy14 - vx7 + vy7 >= 1
 vx16 - vy16 - vx1 + vy1 >= 1
 vx16 - vy16 - vx7 + vy7 >= 1
 vx14 - vy14 - vx7 + vy7 >= 1
 vx14 - vy14 - vx6 + vy6 >= 1
 vx16 - vy16 - vx7 + vy7 >= 1
 vx16 - vy16 - vx6 + vy6 >= 1
 vx14 - vy14 - vx6 + vy6 >= 1
 vx14 - vy14 - vx8 + vy8 >= 1
 vx16 - vy16 - vx6 + vy6 >= 1
 vx16 - vy16 - vx8 + vy8 >= 1
 vx14 - vy14 - vx8 + vy8 >= 1
 vx14 - vy14 - vx11 + vy11 >= 1
 vx16 - vy16 - vx8 + vy8 >= 1
 vx16 - vy16 - vx11 + vy11 >= 1
 vx14 - vy14 - vx11 + vy11 >= 1
 vx14 - vy14 - vx13 + vy13 >= 1
 vx16 - vy16 - vx11 + vy11 >= 1
 vx16 - vy16 - vx13 + vy13 >= 1
 vx14 - vy14 - vx13 + vy13 >= 1
 vx14 - vy14 - vx4 + vy4 >= 1
 vx16 - vy16 - vx13 + vy13 >= 1
 vx16 - vy16 - vx4 + vy4 >= 1
 vx14 - vy14 - vx4 + vy4 >= 1
 vx14 - vy14 - vx36 + vy36 >= 1
 vx16 - vy16 - vx4 + vy4 >= 1
 vx16 - vy16 - vx36 + vy36 >= 1
 vx14 - vy14 - vx36 + vy36 >= 1
 vx14 - vy14 - vx39 + vy39 >= 1
 vx16 - vy16 - vx36 + vy36 >= 1
 vx16 - vy16 - vx39 + vy39 >= 1
 vx14 - vy14 - vx39 + vy39 >= 1
 vx14 - vy14 - vx38 + vy38 >= 1
 vx16 - vy16 - vx39 + vy39 >= 1
 vx16 - vy16 - vx38 + vy38 >= 1
 vx14 + vy14 - vx38 - vy38 >= 1
 vx14 + vy14 - vx37 - vy37 >= 1
 vx16 + vy16 - vx38 - vy38 >= 1
 vx16 + vy16 - vx37 - vy37 >= 1
 vx14 + vy14 - vx17 - vy17 >= 1
 vx14 + vy14 - vx27 - vy27 >= 1
 vx16 + vy16 - vx17 - vy17 >= 1
 vx16 + vy16 - vx27 - vy27 >= 1
 vx14 + vy14 - vx27 - vy27 >= 1
 vx14 + vy14 - vx30 - vy30 >= 1
 vx16 + vy16 - vx27 - vy27 >= 1
 vx16 + vy16 - vx30 - vy30 >= 1
 vx14 + vy14 - vx30 - vy30 >= 1
 vx14 + vy14 - vx31 - vy31 >= 1
 vx16 + vy16 - vx30 - vy30 >= 1
 vx16 + vy16 - vx31 - vy31 >= 1
 vx14 + vy14 - vx31 - vy31 >= 1
 vx14 + vy14 - vx33 - vy33 >= 1
 vx16 + vy16 - vx31 - vy31 >= 1
 vx16 + vy16 - vx33 - vy33 >= 1
 vx14 + vy14 - vx33 - vy33 >= 1
 vx14 + vy14 - vx29 - vy29 >= 1
 vx16 + vy16 - vx33 - vy33 >= 1
 vx16 + vy16 - vx29 - vy29 >= 1
 vx14 + vy14 - vx29 - vy29 >= 1
 vx14 + vy14 - vx34 - vy34 >= 1
 vx16 + vy16 - vx29 - vy29 >= 1
 vx16 + vy16 - vx34 - vy34 >= 1
 vx14 + vy14 - vx34 - vy34 >= 1
 vx14 + vy14 - vx32 - vy32 >= 1
 vx16 + vy16 - vx34 - vy34 >= 1
 vx16 + vy16 - vx32 - vy32 >= 1
 vx14 + vy14 - vx32 - vy32 >= 1
 vx14 + vy14 - vx65 - vy65 >= 1
 vx16 + vy16 - vx32 - vy32 >= 1
 vx16 + vy16 - vx65 - vy65 >= 1
 vx14 + vy14 - vx65 - vy65 >= 1
 vx14 + vy14 - vx28 - vy28 >= 1
 vx16 + vy16 - vx65 - vy65 >= 1
 vx16 + vy16 - vx28 - vy28 >= 1
 vx14 + vy14 - vx28 - vy28 >= 1
 vx14 + vy14 - vx18 - vy18 >= 1
 vx16 + vy16 - vx28 - vy28 >= 1
 vx16 + vy16 - vx18 - vy18 >= 1
 vx14 + vy14 - vx18 - vy18 >= 1
 vx14 + vy14 - vx19 - vy19 >= 1
 vx16 + vy16 - vx18 - vy18 >= 1
 vx16 + vy16 - vx19 - vy19 >= 1
 vx14 + vy14 - vx19 - vy19 >= 1
 vx14 + vy14 - vx26 - vy26 >= 1
 vx16 + vy16 - vx19 - vy19 >= 1
 vx16 + vy16 - vx26 - vy26 >= 1
 vx14 - vx26 >= 1
 vx14 - vx21 >= 1
 vx16 - vx26 >= 1
 vx16 - vx21 >= 1
 vx14 - vy14 - vx21 + vy21 >= 1
 vx14 - vy14 - vx24 + vy24 >= 1
 vx16 - vy16 - vx21 + vy21 >= 1
 vx16 - vy16 - vx24 + vy24 >= 1
 vx14 - vy14 - vx24 + vy24 >= 1
 vx14 - vy14 - vx23 + vy23 >= 1
 vx16 - vy16 - vx24 + vy24 >= 1
 vx16 - vy16 - vx23 + vy23 >= 1
 vx14 - vy14 - vx23 + vy23 >= 1
 vx14 - vy14 - vx20 + vy20 >= 1
 vx16 - vy16 - vx23 + vy23 >= 1
 vx16 - vy16 - vx20 + vy20 >= 1
 vx14 - vy14 - vx20 + vy20 >= 1
 vx14 - vy14 - vx25 + vy25 >= 1
 vx16 - vy16 - vx20 + vy20 >= 1
 vx16 - vy16 - vx25 + vy25 >= 1
 vx14 - vy14 - vx25 + vy25 >= 1
 vx14 - vy14 - vx22 + vy22 >= 1
 vx16 - vy16 - vx25 + vy25 >= 1
 vx16 - vy16 - vx22 + vy22 >= 1
 vx14 - vy14 - vx41 + vy41 <= -1
 vx14 - vy14 - vx64 + vy64 <= -1
 vx16 - vy16 - vx41 + vy41 <= -1
 vx16 - vy16 - vx64 + vy64 <= -1
 vx14 - vy14 - vx64 + vy64 <= -1
 vx14 - vy14 - vx43 + vy43 <= -1
 vx16 - vy16 - vx64 + vy64 <= -1
 vx16 - vy16 - vx43 + vy43 <= -1
 vx14 - vy14 - vx43 + vy43 <= -1
 vx14 - vy14 - vx44 + vy44 <= -1
 vx16 - vy16 - vx43 + vy43 <= -1
 vx16 - vy16 - vx44 + vy44 <= -1
 vx14 - vy14 - vx44 + vy44 <= -1
 vx14 - vy14 - vx3 + vy3 <= -1
 vx16 - vy16 - vx44 + vy44 <= -1
 vx16 - vy16 - vx3 + vy3 <= -1
 vx14 - vy14 - vx3 + vy3 <= -1
 vx14 - vy14 - vx45 + vy45 <= -1
 vx16 - vy16 - vx3 + vy3 <= -1
 vx16 - vy16 - vx45 + vy45 <= -1
 vx14 - vy14 - vx45 + vy45 <= -1
 vx14 - vy14 - vx46 + vy46 <= -1
 vx16 - vy16 - vx45 + vy45 <= -1
 vx16 - vy16 - vx46 + vy46 <= -1
 vy14 - vy46 >= 1
 vy14 - vy47 >= 1
 vy16 - vy46 >= 1
 vy16 - vy47 >= 1
 vx14 + vy14 - vx47 - vy47 >= 1
 vx14 + vy14 - vx48 - vy48 >= 1
 vx16 + vy16 - vx47 - vy47 >= 1
 vx16 + vy16 - vx48 - vy48 >= 1
 vx14 + vy14 - vx48 - vy48 >= 1
 vx14 + vy14 - vx35 - vy35 >= 1
 vx16 + vy16 - vx48 - vy48 >= 1
 vx16 + vy16 - vx35 - vy35 >= 1
 vx14 + vy14 - vx35 - vy35 >= 1
 vx14 + vy14 - vx49 - vy49 >= 1
 vx16 + vy16 - vx35 - vy35 >= 1
 vx16 + vy16 - vx49 - vy49 >= 1
 vx14 + vy14 - vx49 - vy49 >= 1
 vx14 + vy14 - vx50 - vy50 >= 1
 vx16 + vy16 - vx49 - vy49 >= 1
 vx16 + vy16 - vx50 - vy50 >= 1
 vx14 + vy14 - vx50 - vy50 >= 1
 vx14 + vy14 - vx51 - vy51 >= 1
 vx16 + vy16 - vx50 - vy50 >= 1
 vx16 + vy16 - vx51 - vy51 >= 1
 vx14 + vy14 - vx51 - vy51 >= 1
 vx14 + vy14 - vx52 - vy52 >= 1
 vx16 + vy16 - vx51 - vy51 >= 1
 vx16 + vy16 - vx52 - vy52 >= 1
 vx14 - vx52 >= 1
 vx14 - vx62 >= 1
 vx16 - vx52 >= 1
 vx16 - vx62 >= 1
 vx14 - vy14 - vx62 + vy62 >= 1
 vx14 - vy14 - vx53 + vy53 >= 1
 vx16 - vy16 - vx62 + vy62 >= 1
 vx16 - vy16 - vx53 + vy53 >= 1
 vx14 - vy14 - vx53 + vy53 >= 1
 vx14 - vy14 - vx54 + vy54 >= 1
 vx16 - vy16 - vx53 + vy53 >= 1
 vx16 - vy16 - vx54 + vy54 >= 1
 vx14 - vy14 - vx54 + vy54 >= 1
 vx14 - vy14 - vx55 + vy55 >= 1
 vx16 - vy16 - vx54 + vy54 >= 1
 vx16 - vy16 - vx55 + vy55 >= 1
 vx14 - vy14 - vx55 + vy55 >= 1
 vx14 - vy14 - vx56 + vy56 >= 1
 vx16 - vy16 - vx55 + vy55 >= 1
 vx16 - vy16 - vx56 + vy56 >= 1
 vx14 - vy14 - vx56 + vy56 >= 1
 vx14 - vy14 - vx57 + vy57 >= 1
 vx16 - vy16 - vx56 + vy56 >= 1
 vx16 - vy16 - vx57 + vy57 >= 1
 vx14 - vy14 - vx57 + vy57 >= 1
 vx14 - vy14 - vx58 + vy58 >= 1
 vx16 - vy16 - vx57 + vy57 >= 1
 vx16 - vy16 - vx58 + vy58 >= 1
 vx14 - vy14 - vx58 + vy58 >= 1
 vx14 - vy14 - vx59 + vy59 >= 1
 vx16 - vy16 - vx58 + vy58 >= 1
 vx16 - vy16 - vx59 + vy59 >= 1
 vx14 - vy14 - vx59 + vy59 >= 1
 vx14 - vy14 - vx60 + vy60 >= 1
 vx16 - vy16 - vx59 + vy59 >= 1
 vx16 - vy16 - vx60 + vy60 >= 1
 vx14 - vy14 - vx60 + vy60 >= 1
 vx14 - vy14 - vx61 + vy61 >= 1
 vx16 - vy16 - vx60 + vy60 >= 1
 vx16 - vy16 - vx61 + vy61 >= 1
 q8 - oa8 - ob8 - oc8 - od8 = 0
 a8 + a9 - 2 ua8 - oa8 = 0
 b8 + b9 - 2 ub8 - ob8 = 0
 c8 + c9 - 2 uc8 - oc8 = 0
 d8 + d9 - 2 ud8 - od8 = 0
 vx16 - vy16 - vx12 + vy12 >= 1
 vx16 - vy16 - vx9 + vy9 >= 1
 vx10 - vy10 - vx12 + vy12 >= 1
 vx10 - vy10 - vx9 + vy9 >= 1
 vx16 - vy16 - vx9 + vy9 >= 1
 vx16 - vy16 - vx1 + vy1 >= 1
 vx10 - vy10 - vx9 + vy9 >= 1
 vx10 - vy10 - vx1 + vy1 >= 1
 vx16 - vy16 - vx1 + vy1 >= 1
 vx16 - vy16 - vx7 + vy7 >= 1
 vx10 - vy10 - vx1 + vy1 >= 1
 vx10 - vy10 - vx7 + vy7 >= 1
 vx16 - vy16 - vx7 + vy7 >= 1
 vx16 - vy16 - vx6 + vy6 >= 1
 vx10 - vy10 - vx7 + vy7 >= 1
 vx10 - vy10 - vx6 + vy6 >= 1
 vx16 - vy16 - vx6 + vy6 >= 1
 vx16 - vy16 - vx8 + vy8 >= 1
 vx10 - vy10 - vx6 + vy6 >= 1
 vx10 - vy10 - vx8 + vy8 >= 1
 vx16 - vy16 - vx8 + vy8 >= 1
 vx16 - vy16 - vx11 + vy11 >= 1
 vx10 - vy10 - vx8 + vy8 >= 1
 vx10 - vy10 - vx11 + vy11 >= 1
 vx16 - vy16 - vx11 + vy11 >= 1
 vx16 - vy16 - vx13 + vy13 >= 1
 vx10 - vy10 - vx11 + vy11 >= 1
 vx10 - vy10 - vx13 + vy13 >= 1
 vx16 - vy16 - vx13 + vy13 >= 1
 vx16 - vy16 - vx4 + vy4 >= 1
 vx10 - vy10 - vx13 + vy13 >= 1
 vx10 - vy10 - vx4 + vy4 >= 1
 vx16 - vy16 - vx4 + vy4 >= 1
 vx16 - vy16 - vx36 + vy36 >= 1
 vx10 - vy10 - vx4 + vy4 >= 1
 vx10 - vy10 - vx36 + vy36 >= 1
 vx16 - vy16 - vx36 + vy36 >= 1
 vx16 - vy16 - vx39 + vy39 >= 1
 vx10 - vy10 - vx36 + vy36 >= 1
 vx10 - vy10 - vx39 + vy39 >= 1
 vx16 - vy16 - vx39 + vy39 >= 1
 vx16 - vy16 - vx38 + vy38 >= 1
 vx10 - vy10 - vx39 + vy39 >= 1
 vx10 - vy10 - vx38 + vy38 >= 1
 vx16 + vy16 - vx38 - vy38 >= 1
 vx16 + vy16 - vx37 - vy37 >= 1
 vx10 + vy10 - vx38 - vy38 >= 1
 vx10 + vy10 - vx37 - vy37 >= 1
 vx16 + vy16 - vx17 - vy17 >= 1
 vx16 + vy16 - vx27 - vy27 >= 1
 vx10 + vy10 - vx17 - vy17 >= 1
 vx10 + vy10 - vx27 - vy27 >= 1
 vx16 + vy16 - vx27 - vy27 >= 1
 vx16 + vy16 - vx30 - vy30 >= 1
 vx10 + vy10 - vx27 - vy27 >= 1
 vx10 + vy10 - vx30 - vy30 >= 1
 vx16 + vy16 - vx30 - vy30 >= 1
 vx16 + vy16 - vx31 - vy31 >= 1
 vx10 + vy10 - vx30 - vy30 >= 1
 vx10 + vy10 - vx31 - vy31 >= 1
 vx16 + vy16 - vx31 - vy31 >= 1
 vx16 + vy16 - vx33 - vy33 >= 1
 vx10 + vy10 - vx31 - vy31 >= 1
 vx10 + vy10 - vx33 - vy33 >= 1
 vx16 + vy16 - vx33 - vy33 >= 1
 vx16 + vy16 - vx29 - vy29 >= 1
 vx10 + vy10 - vx33 - vy33 >= 1
 vx10 + vy10 - vx29 - vy29 >= 1
 vx16 + vy16 - vx29 - vy29 >= 1
 vx16 + vy16 - vx34 - vy34 >= 1
 vx10 + vy10 - vx29 - vy29 >= 1
 vx10 + vy10 - vx34 - vy34 >= 1
 vx16 + vy16 - vx34 - vy34 >= 1
 vx16 + vy16 - vx32 - vy32 >= 1
 vx10 + vy10 - vx34 - vy34 >= 1
 vx10 + vy10 - vx32 - vy32 >= 1
 vx16 + vy16 - vx32 - vy32 >= 1
 vx16 + vy16 - vx65 - vy65 >= 1
 vx10 + vy10 - vx32 - vy32 >= 1
 vx10 + vy10 - vx65 - vy65 >= 1
 vx16 + vy16 - vx65 - vy65 >= 1
 vx16 + vy16 - vx28 - vy28 >= 1
 vx10 + vy10 - vx65 - vy65 >= 1
 vx10 + vy10 - vx28 - vy28 >= 1
 vx16 + vy16 - vx28 - vy28 >= 1
 vx16 + vy16 - vx18 - vy18 >= 1
 vx10 + vy10 - vx28 - vy28 >= 1
 vx10 + vy10 - vx18 - vy18 >= 1
 vx16 + vy16 - vx18 - vy18 >= 1
 vx16 + vy16 - vx19 - vy19 >= 1
 vx10 + vy10 - vx18 - vy18 >= 1
 vx10 + vy10 - vx19 - vy19 >= 1
 vx16 + vy16 - vx19 - vy19 >= 1
 vx16 + vy16 - vx26 - vy26 >= 1
 vx10 + vy10 - vx19 - vy19 >= 1
 vx10 + vy10 - vx26 - vy26 >= 1
 vx16 - vx26 >= 1
 vx16 - vx21 >= 1
 vx10 - vx26 >= 1
 vx10 - vx21 >= 1
 vx16 - vy16 - vx21 + vy21 >= 1
 vx16 - vy16 - vx24 + vy24 >= 1
 vx10 - vy10 - vx21 + vy21 >= 1
 vx10 - vy10 - vx24 + vy24 >= 1
 vx16 - vy16 - vx24 + vy24 >= 1
 vx16 - vy16 - vx23 + vy23 >= 1
 vx10 - vy10 - vx24 + vy24 >= 1
 vx10 - vy10 - vx23 + vy23 >= 1
 vx16 - vy16 - vx23 + vy23 >= 1
 vx16 - vy16 - vx20 + vy20 >= 1
 vx10 - vy10 - vx23 + vy23 >= 1
 vx10 - vy10 - vx20 + vy20 >= 1
 vx16 - vy16 - vx20 + vy20 >= 1
 vx16 - vy16 - vx25 + vy25 >= 1
 vx10 - vy10 - vx20 + vy20 >= 1
 vx10 - vy10 - vx25 + vy25 >= 1
 vx16 - vy16 - vx25 + vy25 >= 1
 vx16 - vy16 - vx22 + vy22 >= 1
 vx10 - vy10 - vx25 + vy25 >= 1
 vx10 - vy10 - vx22 + vy22 >= 1
 vx16 - vy16 - vx41 + vy41 <= -1
 vx16 - vy16 - vx64 + vy64 <= -1
 vx10 - vy10 - vx41 + vy41 <= -1
 vx10 - vy10 - vx64 + vy64 <= -1
 vx16 - vy16 - vx64 + vy64 <= -1
 vx16 - vy16 - vx43 + vy43 <= -1
 vx10 - vy10 - vx64 + vy64 <= -1
 vx10 - vy10 - vx43 + vy43 <= -1
 vx16 - vy16 - vx43 + vy43 <= -1
 vx16 - vy16 - vx44 + vy44 <= -1
 vx10 - vy10 - vx43 + vy43 <= -1
 vx10 - vy10 - vx44 + vy44 <= -1
 vx16 - vy16 - vx44 + vy44 <= -1
 vx16 - vy16 - vx3 + vy3 <= -1
 vx10 - vy10 - vx44 + vy44 <= -1
 vx10 - vy10 - vx3 + vy3 <= -1
 vx16 - vy16 - vx3 + vy3 <= -1
 vx16 - vy16 - vx45 + vy45 <= -1
 vx10 - vy10 - vx3 + vy3 <= -1
 vx10 - vy10 - vx45 + vy45 <= -1
 vx16 - vy16 - vx45 + vy45 <= -1
 vx16 - vy16 - vx46 + vy46 <= -1
 vx10 - vy10 - vx45 + vy45 <= -1
 vx10 - vy10 - vx46 + vy46 <= -1
 vy16 - vy46 >= 1
 vy16 - vy47 >= 1
 vy10 - vy46 >= 1
 vy10 - vy47 >= 1
 vy16 - vy47 >= 1
 vy16 - vy48 >= 1
 vy10 - vy47 >= 1
 vy10 - vy48 >= 1
 vx16 + vy16 - vx48 - vy48 >= 1
 vx16 + vy16 - vx35 - vy35 >= 1
 vx10 + vy10 - vx48 - vy48 >= 1
 vx10 + vy10 - vx35 - vy35 >= 1
 vx16 + vy16 - vx35 - vy35 >= 1
 vx16 + vy16 - vx49 - vy49 >= 1
 vx10 + vy10 - vx35 - vy35 >= 1
 vx10 + vy10 - vx49 - vy49 >= 1
 vx16 + vy16 - vx49 - vy49 >= 1
 vx16 + vy16 - vx50 - vy50 >= 1
 vx10 + vy10 - vx49 - vy49 >= 1
 vx10 + vy10 - vx50 - vy50 >= 1
 vx16 + vy16 - vx50 - vy50 >= 1
 vx16 + vy16 - vx51 - vy51 >= 1
 vx10 + vy10 - vx50 - vy50 >= 1
 vx10 + vy10 - vx51 - vy51 >= 1
 vx16 + vy16 - vx51 - vy51 >= 1
 vx16 + vy16 - vx52 - vy52 >= 1
 vx10 + vy10 - vx51 - vy51 >= 1
 vx10 + vy10 - vx52 - vy52 >= 1
 vx16 - vx52 >= 1
 vx16 - vx62 >= 1
 vx10 - vx52 >= 1
 vx10 - vx62 >= 1
 vx16 - vy16 - vx62 + vy62 >= 1
 vx16 - vy16 - vx53 + vy53 >= 1
 vx10 - vy10 - vx62 + vy62 >= 1
 vx10 - vy10 - vx53 + vy53 >= 1
 vx16 - vy16 - vx53 + vy53 >= 1
 vx16 - vy16 - vx54 + vy54 >= 1
 vx10 - vy10 - vx53 + vy53 >= 1
 vx10 - vy10 - vx54 + vy54 >= 1
 vx16 - vy16 - vx54 + vy54 >= 1
 vx16 - vy16 - vx55 + vy55 >= 1
 vx10 - vy10 - vx54 + vy54 >= 1
 vx10 - vy10 - vx55 + vy55 >= 1
 vx16 - vy16 - vx55 + vy55 >= 1
 vx16 - vy16 - vx56 + vy56 >= 1
 vx10 - vy10 - vx55 + vy55 >= 1
 vx10 - vy10 - vx56 + vy56 >= 1
 vx16 - vy16 - vx56 + vy56 >= 1
 vx16 - vy16 - vx57 + vy57 >= 1
 vx10 - vy10 - vx56 + vy56 >= 1
 vx10 - vy10 - vx57 + vy57 >= 1
 vx16 - vy16 - vx57 + vy57 >= 1
 vx16 - vy16 - vx58 + vy58 >= 1
 vx10 - vy10 - vx57 + vy57 >= 1
 vx10 - vy10 - vx58 + vy58 >= 1
 vx16 - vy16 - vx58 + vy58 >= 1
 vx16 - vy16 - vx59 + vy59 >= 1
 vx10 - vy10 - vx58 + vy58 >= 1
 vx10 - vy10 - vx59 + vy59 >= 1
 vx16 - vy16 - vx59 + vy59 >= 1
 vx16 - vy16 - vx60 + vy60 >= 1
 vx10 - vy10 - vx59 + vy59 >= 1
 vx10 - vy10 - vx60 + vy60 >= 1
 vx16 - vy16 - vx60 + vy60 >= 1
 vx16 - vy16 - vx61 + vy61 >= 1
 vx10 - vy10 - vx60 + vy60 >= 1
 vx10 - vy10 - vx61 + vy61 >= 1
 q9 - oa9 - ob9 - oc9 - od9 = 0
 a9 + a10 - 2 ua9 - oa9 = 0
 b9 + b10 - 2 ub9 - ob9 = 0
 c9 + c10 - 2 uc9 - oc9 = 0
 d9 + d10 - 2 ud9 - od9 = 0
 vx10 - vy10 - vx9 + vy9 >= 1
 vx10 - vy10 - vx1 + vy1 >= 1
 vx12 - vy12 - vx9 + vy9 >= 1
 vx12 - vy12 - vx1 + vy1 >= 1
 vx10 - vy10 - vx1 + vy1 >= 1
 vx10 - vy10 - vx7 + vy7 >= 1
 vx12 - vy12 - vx1 + vy1 >= 1
 vx12 - vy12 - vx7 + vy7 >= 1
 vx10 - vy10 - vx7 + vy7 >= 1
 vx10 - vy10 - vx6 + vy6 >= 1
 vx12 - vy12 - vx7 + vy7 >= 1
 vx12 - vy12 - vx6 + vy6 >= 1
 vx10 - vy10 - vx6 + vy6 >= 1
 vx10 - vy10 - vx8 + vy8 >= 1
 vx12 - vy12 - vx6 + vy6 >= 1
 vx12 - vy12 - vx8 + vy8 >= 1
 vx10 - vy10 - vx8 + vy8 >= 1
 vx10 - vy10 - vx11 + vy11 >= 1
 vx12 - vy12 - vx8 + vy8 >= 1
 vx12 - vy12 - vx11 + vy11 >= 1
 vx10 - vy10 - vx11 + vy11 >= 1
 vx10 - vy10 - vx13 + vy13 >= 1
 vx12 - vy12 - vx11 + vy11 >= 1
 vx12 - vy12 - vx13 + vy13 >= 1
 vx10 - vy10 - vx13 + vy13 >= 1
 vx10 - vy10 - vx4 + vy4 >= 1
 vx12 - vy12 - vx13 + vy13 >= 1
 vx12 - vy12 - vx4 + vy4 >= 1
 vx10 - vy10 - vx4 + vy4 >= 1
 vx10 - vy10 - vx36 + vy36 >= 1
 vx12 - vy12 - vx4 + vy4 >= 1
 vx12 - vy12 - vx36 + vy36 >= 1
 vx10 - vy10 - vx36 + vy36 >= 1
 vx10 - vy10 - vx39 + vy39 >= 1
 vx12 - vy12 - vx36 + vy36 >= 1
 vx12 - vy12 - vx39 + vy39 >= 1
 vx10 - vy10 - vx39 + vy39 >= 1
 vx10 - vy10 - vx38 + vy38 >= 1
 vx12 - vy12 - vx39 + vy39 >= 1
 vx12 - vy12 - vx38 + vy38 >= 1
 vx10 + vy10 - vx38 - vy38 >= 1
 vx10 + vy10 - vx37 - vy37 >= 1
 vx12 + vy12 - vx38 - vy38 >= 1
 vx12 + vy12 - vx37 - vy37 >= 1
 vx10 + vy10 - vx17 - vy17 >= 1
 vx10 + vy10 - vx27 - vy27 >= 1
 vx12 + vy12 - vx17 - vy17 >= 1
 vx12 + vy12 - vx27 - vy27 >= 1
 vx10 + vy10 - vx27 - vy27 >= 1
 vx10 + vy10 - vx30 - vy30 >= 1
 vx12 + vy12 - vx27 - vy27 >= 1
 vx12 + vy12 - vx30 - vy30 >= 1
 vx10 + vy10 - vx30 - vy30 >= 1
 vx10 + vy10 - vx31 - vy31 >= 1
 vx12 + vy12 - vx30 - vy30 >= 1
 vx12 + vy12 - vx31 - vy31 >= 1
 vx10 + vy10 - vx31 - vy31 >= 1
 vx10 + vy10 - vx33 - vy33 >= 1
 vx12 + vy12 - vx31 - vy31 >= 1
 vx12 + vy12 - vx33 - vy33 >= 1
 vx10 + vy10 - vx33 - vy33 >= 1
 vx10 + vy10 - vx29 - vy29 >= 1
 vx12 + vy12 - vx33 - vy33 >= 1
 vx12 + vy12 - vx29 - vy29 >= 1
 vx10 + vy10 - vx29 - vy29 >= 1
 vx10 + vy10 - vx34 - vy34 >= 1
 vx12 + vy12 - vx29 - vy29 >= 1
 vx12 + vy12 - vx34 - vy34 >= 1
 vx10 + vy10 - vx34 - vy34 >= 1
 vx10 + vy10 - vx32 - vy32 >= 1
 vx12 + vy12 - vx34 - vy34 >= 1
 vx12 + vy12 - vx32 - vy32 >= 1
 vx10 + vy10 - vx32 - vy32 >= 1
 vx10 + vy10 - vx65 - vy65 >= 1
 vx12 + vy12 - vx32 - vy32 >= 1
 vx12 + vy12 - vx65 - vy65 >= 1
 vx10 + vy10 - vx65 - vy65 >= 1
 vx10 + vy10 - vx28 - vy28 >= 1
 vx12 + vy12 - vx65 - vy65 >= 1
 vx12 + vy12 - vx28 - vy28 >= 1
 vx10 + vy10 - vx28 - vy28 >= 1
 vx10 + vy10 - vx18 - vy18 >= 1
 vx12 + vy12 - vx28 - vy28 >= 1
 vx12 + vy12 - vx18 - vy18 >= 1
 vx10 + vy10 - vx18 - vy18 >= 1
 vx10 + vy10 - vx19 - vy19 >= 1
 vx12 + vy12 - vx18 - vy18 >= 1
 vx12 + vy12 - vx19 - vy19 >= 1
 vx10 + vy10 - vx19 - vy19 >= 1
 vx10 + vy10 - vx26 - vy26 >= 1
 vx12 + vy12 - vx19 - vy19 >= 1
 vx12 + vy12 - vx26 - vy26 >= 1
 vx10 - vx26 >= 1
 vx10 - vx21 >= 1
 vx12 - vx26 >= 1
 vx12 - vx21 >= 1
 vx10 - vy10 - vx21 + vy21 >= 1
 vx10 - vy10 - vx24 + vy24 >= 1
 vx12 - vy12 - vx21 + vy21 >= 1
 vx12 - vy12 - vx24 + vy24 >= 1
 vx10 - vy10 - vx24 + vy24 >= 1
 vx10 - vy10 - vx23 + vy23 >= 1
 vx12 - vy12 - vx24 + vy24 >= 1
 vx12 - vy12 - vx23 + vy23 >= 1
 vx10 - vy10 - vx23 + vy23 >= 1
 vx10 - vy10 - vx20 + vy20 >= 1
 vx12 - vy12 - vx23 + vy23 >= 1
 vx12 - vy12 - vx20 + vy20 >= 1
 vx10 - vy10 - vx20 + vy20 >= 1
 vx10 - vy10 - vx25 + vy25 >= 1
 vx12 - vy12 - vx20 + vy20 >= 1
 vx12 - vy12 - vx25 + vy25 >= 1
 vx10 - vy10 - vx25 + vy25 >= 1
 vx10 - vy10 - vx22 + vy22 >= 1
 vx12 - vy12 - vx25 + vy25 >= 1
 vx12 - vy12 - vx22 + vy22 >= 1
 vx10 - vy10 - vx41 + vy41 <= -1
 vx10 - vy10 - vx64 + vy64 <= -1
 vx12 - vy12 - vx41 + vy41 <= -1
 vx12 - vy12 - vx64 + vy64 <= -1
 vx10 - vy10 - vx64 + vy64 <= -1
 vx10 - vy10 - vx43 + vy43 <= -1
 vx12 - vy12 - vx64 + vy64 <= -1
 vx12 - vy12 - vx43 + vy43 <= -1
 vx10 - vy10 - vx43 + vy43 <= -1
 vx10 - vy10 - vx44 + vy44 <= -1
 vx12 - vy12 - vx43 + vy43 <= -1
 vx12 - vy12 - vx44 + vy44 <= -1
 vx10 - vy10 - vx44 + vy44 <= -1
 vx10 - vy10 - vx3 + vy3 <= -1
 vx12 - vy12 - vx44 + vy44 <= -1
 vx12 - vy12 - vx3 + vy3 <= -1
 vx10 - vy10 - vx3 + vy3 <= -1
 vx10 - vy10 - vx45 + vy45 <= -1
 vx12 - vy12 - vx3 + vy3 <= -1
 vx12 - vy12 - vx45 + vy45 <= -1
 vx10 - vy10 - vx45 + vy45 <= -1
 vx10 - vy10 - vx46 + vy46 <= -1
 vx12 - vy12 - vx45 + vy45 <= -1
 vx12 - vy12 - vx46 + vy46 <= -1
 vx10 - vy10 - vx46 + vy46 <= -1
 vx10 - vy10 - vx47 + vy47 <= -1
 vx12 - vy12 - vx46 + vy46 <= -1
 vx12 - vy12 - vx47 + vy47 <= -1
 vy10 - vy47 >= 1
 vy10 - vy48 >= 1
 vy12 - vy47 >= 1
 vy12 - vy48 >= 1
 vy10 - vy48 >= 1
 vy10 - vy35 >= 1
 vy12 - vy48 >= 1
 vy12 - vy35 >= 1
 vx10 + vy10 - vx35 - vy35 >= 1
 vx10 + vy10 - vx49 - vy49 >= 1
 vx12 + vy12 - vx35 - vy35 >= 1
 vx12 + vy12 - vx49 - vy49 >= 1
 vx10 + vy10 - vx49 - vy49 >= 1
 vx10 + vy10 - vx50 - vy50 >= 1
 vx12 + vy12 - vx49 - vy49 >= 1
 vx12 + vy12 - vx50 - vy50 >= 1
 vx10 + vy10 - vx50 - vy50 >= 1
 vx10 + vy10 - vx51 - vy51 >= 1
 vx12 + vy12 - vx50 - vy50 >= 1
 vx12 + vy12 - vx51 - vy51 >= 1
 vx10 + vy10 - vx51 - vy51 >= 1
 vx10 + vy10 - vx52 - vy52 >= 1
 vx12 + vy12 - vx51 - vy51 >= 1
 vx12 + vy12 - vx52 - vy52 >= 1
 vx10 - vx52 >= 1
 vx10 - vx62 >= 1
 vx12 - vx52 >= 1
 vx12 - vx62 >= 1
 vx10 - vy10 - vx62 + vy62 >= 1
 vx10 - vy10 - vx53 + vy53 >= 1
 vx12 - vy12 - vx62 + vy62 >= 1
 vx12 - vy12 - vx53 + vy53 >= 1
 vx10 - vy10 - vx53 + vy53 >= 1
 vx10 - vy10 - vx54 + vy54 >= 1
 vx12 - vy12 - vx53 + vy53 >= 1
 vx12 - vy12 - vx54 + vy54 >= 1
 vx10 - vy10 - vx54 + vy54 >= 1
 vx10 - vy10 - vx55 + vy55 >= 1
 vx12 - vy12 - vx54 + vy54 >= 1
 vx12 - vy12 - vx55 + vy55 >= 1
 vx10 - vy10 - vx55 + vy55 >= 1
 vx10 - vy10 - vx56 + vy56 >= 1
 vx12 - vy12 - vx55 + vy55 >= 1
 vx12 - vy12 - vx56 + vy56 >= 1
 vx10 - vy10 - vx56 + vy56 >= 1
 vx10 - vy10 - vx57 + vy57 >= 1
 vx12 - vy12 - vx56 + vy56 >= 1
 vx12 - vy12 - vx57 + vy57 >= 1
 vx10 - vy10 - vx57 + vy57 >= 1
 vx10 - vy10 - vx58 + vy58 >= 1
 vx12 - vy12 - vx57 + vy57 >= 1
 vx12 - vy12 - vx58 + vy58 >= 1
 vx10 - vy10 - vx58 + vy58 >= 1
 vx10 - vy10 - vx59 + vy59 >= 1
 vx12 - vy12 - vx58 + vy58 >= 1
 vx12 - vy12 - vx59 + vy59 >= 1
 vx10 - vy10 - vx59 + vy59 >= 1
 vx10 - vy10 - vx60 + vy60 >= 1
 vx12 - vy12 - vx59 + vy59 >= 1
 vx12 - vy12 - vx60 + vy60 >= 1
 vx10 - vy10 - vx60 + vy60 >= 1
 vx10 - vy10 - vx61 + vy61 >= 1
 vx12 - vy12 - vx60 + vy60 >= 1
 vx12 - vy12 - vx61 + vy61 >= 1
 q10 - oa10 - ob10 - oc10 - od10 = 0
 a10 + a11 - 2 ua10 - oa10 = 0
 b10 + b11 - 2 ub10 - ob10 = 0
 c10 + c11 - 2 uc10 - oc10 = 0
 d10 + d11 - 2 ud10 - od10 = 0
 vx12 - vy12 - vx1 + vy1 >= 1
 vx12 - vy12 - vx7 + vy7 >= 1
 vx9 - vy9 - vx1 + vy1 >= 1
 vx9 - vy9 - vx7 + vy7 >= 1
 vx12 - vy12 - vx7 + vy7 >= 1
 vx12 - vy12 - vx6 + vy6 >= 1
 vx9 - vy9 - vx7 + vy7 >= 1
 vx9 - vy9 - vx6 + vy6 >= 1
 vx12 - vy12 - vx6 + vy6 >= 1
 vx12 - vy12 - vx8 + vy8 >= 1
 vx9 - vy9 - vx6 + vy6 >= 1
 vx9 - vy9 - vx8 + vy8 >= 1
 vx12 - vy12 - vx8 + vy8 >= 1
 vx12 - vy12 - vx11 + vy11 >= 1
 vx9 - vy9 - vx8 + vy8 >= 1
 vx9 - vy9 - vx11 + vy11 >= 1
 vx12 - vy12 - vx11 + vy11 >= 1
 vx12 - vy12 - vx13 + vy13 >= 1
 vx9 - vy9 - vx11 + vy11 >= 1
 vx9 - vy9 - vx13 + vy13 >= 1
 vx12 - vy12 - vx13 + vy13 >= 1
 vx12 - vy12 - vx4 + vy4 >= 1
 vx9 - vy9 - vx13 + vy13 >= 1
 vx9 - vy9 - vx4 + vy4 >= 1
 vx12 - vy12 - vx4 + vy4 >= 1
 vx12 - vy12 - vx36 + vy36 >= 1
 vx9 - vy9 - vx4 + vy4 >= 1
 vx9 - vy9 - vx36 + vy36 >= 1
 vx12 - vy12 - vx36 + vy36 >= 1
 vx12 - vy12 - vx39 + vy39 >= 1
 vx9 - vy9 - vx36 + vy36 >= 1
 vx9 - vy9 - vx39 + vy39 >= 1
 vx12 - vy12 - vx39 + vy39 >= 1
 vx12 - vy12 - vx38 + vy38 >= 1
 vx9 - vy9 - vx39 + vy39 >= 1
 vx9 - vy9 - vx38 + vy38 >= 1
 vx12 + vy12 - vx38 - vy38 >= 1
 vx12 + vy12 - vx37 - vy37 >= 1
 vx9 + vy9 - vx38 - vy38 >= 1
 vx9 + vy9 - vx37 - vy37 >= 1
 vy12 - vy17 >= 1
 vy12 - vy27 >= 1
 vy9 - vy17 >= 1
 vy9 - vy27 >= 1
 vy12 - vy27 >= 1
 vy12 - vy30 >= 1
 vy9 - vy27 >= 1
 vy9 - vy30 >= 1
 vy12 - vy30 >= 1
 vy12 - vy31 >= 1
 vy9 - vy30 >= 1
 vy9 - vy31 >= 1
 vy12 - vy31 >= 1
 vy12 - vy33 >= 1
 vy9 - vy31 >= 1
 vy9 - vy33 >= 1
 vy12 - vy33 >= 1
 vy12 - vy29 >= 1
 vy9 - vy33 >= 1
 vy9 - vy29 >= 1
 vy12 - vy29 >= 1
 vy12 - vy34 >= 1
 vy9 - vy29 >= 1
 vy9 - vy34 >= 1
 vy12 - vy34 >= 1
 vy12 - vy32 >= 1
 vy9 - vy34 >= 1
 vy9 - vy32 >= 1
 vy12 - vy32 >= 1
 vy12 - vy65 >= 1
 vy9 - vy32 >= 1
 vy9 - vy65 >= 1
 vy12 - vy65 >= 1
 vy12 - vy28 >= 1
 vy9 - vy65 >= 1
 vy9 - vy28 >= 1
 vx12 + vy12 - vx28 - vy28 >= 1
 vx12 + vy12 - vx18 - vy18 >= 1
 vx9 + vy9 - vx28 - vy28 >= 1
 vx9 + vy9 - vx18 - vy18 >= 1
 vx12 + vy12 - vx18 - vy18 >= 1
 vx12 + vy12 - vx19 - vy19 >= 1
 vx9 + vy9 - vx18 - vy18 >= 1
 vx9 + vy9 - vx19 - vy19 >= 1
 vx12 + vy12 - vx19 - vy19 >= 1
 vx12 + vy12 - vx26 - vy26 >= 1
 vx9 + vy9 - vx19 - vy19 >= 1
 vx9 + vy9 - vx26 - vy26 >= 1
 vx12 - vx26 >= 1
 vx12 - vx21 >= 1
 vx9 - vx26 >= 1
 vx9 - vx21 >= 1
 vx12 - vy12 - vx21 + vy21 >= 1
 vx12 - vy12 - vx24 + vy24 >= 1
 vx9 - vy9 - vx21 + vy21 >= 1
 vx9 - vy9 - vx24 + vy24 >= 1
 vx12 - vy12 - vx24 + vy24 >= 1
 vx12 - vy12 - vx23 + vy23 >= 1
 vx9 - vy9 - vx24 + vy24 >= 1
 vx9 - vy9 - vx23 + vy23 >= 1
 vx12 - vy12 - vx23 + vy23 >= 1
 vx12 - vy12 - vx20 + vy20 >= 1
 vx9 - vy9 - vx23 + vy23 >= 1
 vx9 - vy9 - vx20 + vy20 >= 1
 vx12 - vy12 - vx20 + vy20 >= 1
 vx12 - vy12 - vx25 + vy25 >= 1
 vx9 - vy9 - vx20 + vy20 >= 1
 vx9 - vy9 - vx25 + vy25 >= 1
 vy12 - vy25 <= -1
 vy12 - vy22 <= -1
 vy9 - vy25 <= -1
 vy9 - vy22 <= -1
 vx12 - vy12 - vx41 + vy41 <= -1
 vx12 - vy12 - vx64 + vy64 <= -1
 vx9 - vy9 - vx41 + vy41 <= -1
 vx9 - vy9 - vx64 + vy64 <= -1
 vx12 - vy12 - vx64 + vy64 <= -1
 vx12 - vy12 - vx43 + vy43 <= -1
 vx9 - vy9 - vx64 + vy64 <= -1
 vx9 - vy9 - vx43 + vy43 <= -1
 vx12 - vy12 - vx43 + vy43 <= -1
 vx12 - vy12 - vx44 + vy44 <= -1
 vx9 - vy9 - vx43 + vy43 <= -1
 vx9 - vy9 - vx44 + vy44 <= -1
 vx12 - vy12 - vx44 + vy44 <= -1
 vx12 - vy12 - vx3 + vy3 <= -1
 vx9 - vy9 - vx44 + vy44 <= -1
 vx9 - vy9 - vx3 + vy3 <= -1
 vx12 - vy12 - vx3 + vy3 <= -1
 vx12 - vy12 - vx45 + vy45 <= -1
 vx9 - vy9 - vx3 + vy3 <= -1
 vx9 - vy9 - vx45 + vy45 <= -1
 vx12 - vy12 - vx45 + vy45 <= -1
 vx12 - vy12 - vx46 + vy46 <= -1
 vx9 - vy9 - vx45 + vy45 <= -1
 vx9 - vy9 - vx46 + vy46 <= -1
 vx12 - vy12 - vx46 + vy46 <= -1
 vx12 - vy12 - vx47 + vy47 <= -1
 vx9 - vy9 - vx46 + vy46 <= -1
 vx9 - vy9 - vx47 + vy47 <= -1
 vx12 - vy12 - vx47 + vy47 <= -1
 vx12 - vy12 - vx48 + vy48 <= -1
 vx9 - vy9 - vx47 + vy47 <= -1
 vx9 - vy9 - vx48 + vy48 <= -1
 vy12 - vy48 >= 1
 vy12 - vy35 >= 1
 vy9 - vy48 >= 1
 vy9 - vy35 >= 1
 vx12 + vy12 - vx35 - vy35 >= 1
 vx12 + vy12 - vx49 - vy49 >= 1
 vx9 + vy9 - vx35 - vy35 >= 1
 vx9 + vy9 - vx49 - vy49 >= 1
 vx12 + vy12 - vx49 - vy49 >= 1
 vx12 + vy12 - vx50 - vy50 >= 1
 vx9 + vy9 - vx49 - vy49 >= 1
 vx9 + vy9 - vx50 - vy50 >= 1
 vx12 + vy12 - vx50 - vy50 >= 1
 vx12 + vy12 - vx51 - vy51 >= 1
 vx9 + vy9 - vx50 - vy50 >= 1
 vx9 + vy9 - vx51 - vy51 >= 1
 vx12 + vy12 - vx51 - vy51 >= 1
 vx12 + vy12 - vx52 - vy52 >= 1
 vx9 + vy9 - vx51 - vy51 >= 1
 vx9 + vy9 - vx52 - vy52 >= 1
 vx12 - vx52 >= 1
 vx12 - vx62 >= 1
 vx9 - vx52 >= 1
 vx9 - vx62 >= 1
 vx12 - vy12 - vx62 + vy62 >= 1
 vx12 - vy12 - vx53 + vy53 >= 1
 vx9 - vy9 - vx62 + vy62 >= 1
 vx9 - vy9 - vx53 + vy53 >= 1
 vx12 - vy12 - vx53 + vy53 >= 1
 vx12 - vy12 - vx54 + vy54 >= 1
 vx9 - vy9 - vx53 + vy53 >= 1
 vx9 - vy9 - vx54 + vy54 >= 1
 vx12 - vy12 - vx54 + vy54 >= 1
 vx12 - vy12 - vx55 + vy55 >= 1
 vx9 - vy9 - vx54 + vy54 >= 1
 vx9 - vy9 - vx55 + vy55 >= 1
 vx12 - vy12 - vx55 + vy55 >= 1
 vx12 - vy12 - vx56 + vy56 >= 1
 vx9 - vy9 - vx55 + vy55 >= 1
 vx9 - vy9 - vx56 + vy56 >= 1
 vx12 - vy12 - vx56 + vy56 >= 1
 vx12 - vy12 - vx57 + vy57 >= 1
 vx9 - vy9 - vx56 + vy56 >= 1
 vx9 - vy9 - vx57 + vy57 >= 1
 vx12 - vy12 - vx57 + vy57 >= 1
 vx12 - vy12 - vx58 + vy58 >= 1
 vx9 - vy9 - vx57 + vy57 >= 1
 vx9 - vy9 - vx58 + vy58 >= 1
 vx12 - vy12 - vx58 + vy58 >= 1
 vx12 - vy12 - vx59 + vy59 >= 1
 vx9 - vy9 - vx58 + vy58 >= 1
 vx9 - vy9 - vx59 + vy59 >= 1
 vx12 - vy12 - vx59 + vy59 >= 1
 vx12 - vy12 - vx60 + vy60 >= 1
 vx9 - vy9 - vx59 + vy59 >= 1
 vx9 - vy9 - vx60 + vy60 >= 1
 vx12 - vy12 - vx60 + vy60 >= 1
 vx12 - vy12 - vx61 + vy61 >= 1
 vx9 - vy9 - vx60 + vy60 >= 1
 vx9 - vy9 - vx61 + vy61 >= 1
 q11 - oa11 - ob11 - oc11 - od11 = 0
 a11 + a12 - 2 ua11 - oa11 = 0
 b11 + b12 - 2 ub11 - ob11 = 0
 c11 + c12 - 2 uc11 - oc11 = 0
 d11 + d12 - 2 ud11 - od11 = 0
 vx9 - vy9 - vx7 + vy7 >= 1
 vx9 - vy9 - vx6 + vy6 >= 1
 vx1 - vy1 - vx7 + vy7 >= 1
 vx1 - vy1 - vx6 + vy6 >= 1
 vx9 - vy9 - vx6 + vy6 >= 1
 vx9 - vy9 - vx8 + vy8 >= 1
 vx1 - vy1 - vx6 + vy6 >= 1
 vx1 - vy1 - vx8 + vy8 >= 1
 vx9 - vy9 - vx8 + vy8 >= 1
 vx9 - vy9 - vx11 + vy11 >= 1
 vx1 - vy1 - vx8 + vy8 >= 1
 vx1 - vy1 - vx11 + vy11 >= 1
 vx9 - vy9 - vx11 + vy11 >= 1
 vx9 - vy9 - vx13 + vy13 >= 1
 vx1 - vy1 - vx11 + vy11 >= 1
 vx1 - vy1 - vx13 + vy13 >= 1
 vx9 - vy9 - vx13 + vy13 >= 1
 vx9 - vy9 - vx4 + vy4 >= 1
 vx1 - vy1 - vx13 + vy13 >= 1
 vx1 - vy1 - vx4 + vy4 >= 1
 vx9 - vy9 - vx4 + vy4 >= 1
 vx9 - vy9 - vx36 + vy36 >= 1
 vx1 - vy1 - vx4 + vy4 >= 1
 vx1 - vy1 - vx36 + vy36 >= 1
 vx9 - vy9 - vx36 + vy36 >= 1
 vx9 - vy9 - vx39 + vy39 >= 1
 vx1 - vy1 - vx36 + vy36 >= 1
 vx1 - vy1 - vx39 + vy39 >= 1
 vx9 - vy9 - vx39 + vy39 >= 1
 vx9 - vy9 - vx38 + vy38 >= 1
 vx1 - vy1 - vx39 + vy39 >= 1
 vx1 - vy1 - vx38 + vy38 >= 1
 vx9 + vy9 - vx38 - vy38 >= 1
 vx9 + vy9 - vx37 - vy37 >= 1
 vx1 + vy1 - vx38 - vy38 >= 1
 vx1 + vy1 - vx37 - vy37 >= 1
 vx9 - vy9 - vx17 + vy17 <= -1
 vx9 - vy9 - vx27 + vy27 <= -1
 vx1 - vy1 - vx17 + vy17 <= -1
 vx1 - vy1 - vx27 + vy27 <= -1
 vx9 - vy9 - vx27 + vy27 <= -1
 vx9 - vy9 - vx30 + vy30 <= -1
 vx1 - vy1 - vx27 + vy27 <= -1
 vx1 - vy1 - vx30 + vy30 <= -1
 vx9 - vy9 - vx30 + vy30 <= -1
 vx9 - vy9 - vx31 + vy31 <= -1
 vx1 - vy1 - vx30 + vy30 <= -1
 vx1 - vy1 - vx31 + vy31 <= -1
 vx9 - vy9 - vx31 + vy31 <= -1
 vx9 - vy9 - vx33 + vy33 <= -1
 vx1 - vy1 - vx31 + vy31 <= -1
 vx1 - vy1 - vx33 + vy33 <= -1
 vx9 - vy9 - vx33 + vy33 <= -1
 vx9 - vy9 - vx29 + vy29 <= -1
 vx1 - vy1 - vx33 + vy33 <= -1
 vx1 - vy1 - vx29 + vy29 <= -1
 vx9 - vy9 - vx29 + vy29 <= -1
 vx9 - vy9 - vx34 + vy34 <= -1
 vx1 - vy1 - vx29 + vy29 <= -1
 vx1 - vy1 - vx34 + vy34 <= -1
 vx9 - vy9 - vx34 + vy34 <= -1
 vx9 - vy9 - vx32 + vy32 <= -1
 vx1 - vy1 - vx34 + vy34 <= -1
 vx1 - vy1 - vx32 + vy32 <= -1
 vx9 - vy9 - vx32 + vy32 <= -1
 vx9 - vy9 - vx65 + vy65 <= -1
 vx1 - vy1 - vx32 + vy32 <= -1
 vx1 - vy1 - vx65 + vy65 <= -1
 vx9 - vy9 - vx65 + vy65 <= -1
 vx9 - vy9 - vx28 + vy28 <= -1
 vx1 - vy1 - vx65 + vy65 <= -1
 vx1 - vy1 - vx28 + vy28 <= -1
 vy9 - vy28 >= 1
 vy9 - vy18 >= 1
 vy1 - vy28 >= 1
 vy1 - vy18 >= 1
 vx9 + vy9 - vx18 - vy18 >= 1
 vx9 + vy9 - vx19 - vy19 >= 1
 vx1 + vy1 - vx18 - vy18 >= 1
 vx1 + vy1 - vx19 - vy19 >= 1
 vx9 + vy9 - vx19 - vy19 >= 1
 vx9 + vy9 - vx26 - vy26 >= 1
 vx1 + vy1 - vx19 - vy19 >= 1
 vx1 + vy1 - vx26 - vy26 >= 1
 vy9 - vy21 <= -1
 vy9 - vy24 <= -1
 vy1 - vy21 <= -1
 vy1 - vy24 <= -1
 vy9 - vy24 <= -1
 vy9 - vy23 <= -1
 vy1 - vy24 <= -1
 vy1 - vy23 <= -1
 vy9 - vy23 <= -1
 vy9 - vy20 <= -1
 vy1 - vy23 <= -1
 vy1 - vy20 <= -1
 vy9 - vy20 <= -1
 vy9 - vy25 <= -1
 vy1 - vy20 <= -1
 vy1 - vy25 <= -1
 vx9 + vy9 - vx25 - vy25 <= -1
 vx9 + vy9 - vx22 - vy22 <= -1
 vx1 + vy1 - vx25 - vy25 <= -1
 vx1 + vy1 - vx22 - vy22 <= -1
 vx9 - vy9 - vx41 + vy41 <= -1
 vx9 - vy9 - vx64 + vy64 <= -1
 vx1 - vy1 - vx41 + vy41 <= -1
 vx1 - vy1 - vx64 + vy64 <= -1
 vx9 - vy9 - vx64 + vy64 <= -1
 vx9 - vy9 - vx43 + vy43 <= -1
 vx1 - vy1 - vx64 + vy64 <= -1
 vx1 - vy1 - vx43 + vy43 <= -1
 vx9 - vy9 - vx43 + vy43 <= -1
 vx9 - vy9 - vx44 + vy44 <= -1
 vx1 - vy1 - vx43 + vy43 <= -1
 vx1 - vy1 - vx44 + vy44 <= -1
 vx9 - vy9 - vx44 + vy44 <= -1
 vx9 - vy9 - vx3 + vy3 <= -1
 vx1 - vy1 - vx44 + vy44 <= -1
 vx1 - vy1 - vx3 + vy3 <= -1
 vx9 - vy9 - vx3 + vy3 <= -1
 vx9 - vy9 - vx45 + vy45 <= -1
 vx1 - vy1 - vx3 + vy3 <= -1
 vx1 - vy1 - vx45 + vy45 <= -1
 vx9 - vy9 - vx45 + vy45 <= -1
 vx9 - vy9 - vx46 + vy46 <= -1
 vx1 - vy1 - vx45 + vy45 <= -1
 vx1 - vy1 - vx46 + vy46 <= -1
 vx9 - vy9 - vx46 + vy46 <= -1
 vx9 - vy9 - vx47 + vy47 <= -1
 vx1 - vy1 - vx46 + vy46 <= -1
 vx1 - vy1 - vx47 + vy47 <= -1
 vx9 - vy9 - vx47 + vy47 <= -1
 vx9 - vy9 - vx48 + vy48 <= -1
 vx1 - vy1 - vx47 + vy47 <= -1
 vx1 - vy1 - vx48 + vy48 <= -1
 vx9 - vy9 - vx48 + vy48 <= -1
 vx9 - vy9 - vx35 + vy35 <= -1
 vx1 - vy1 - vx48 + vy48 <= -1
 vx1 - vy1 - vx35 + vy35 <= -1
 vy9 - vy35 >= 1
 vy9 - vy49 >= 1
 vy1 - vy35 >= 1
 vy1 - vy49 >= 1
 vx9 + vy9 - vx49 - vy49 >= 1
 vx9 + vy9 - vx50 - vy50 >= 1
 vx1 + vy1 - vx49 - vy49 >= 1
 vx1 + vy1 - vx50 - vy50 >= 1
 vx9 + vy9 - vx50 - vy50 >= 1
 vx9 + vy9 - vx51 - vy51 >= 1
 vx1 + vy1 - vx50 - vy50 >= 1
 vx1 + vy1 - vx51 - vy51 >= 1
 vx9 + vy9 - vx51 - vy51 >= 1
 vx9 + vy9 - vx52 - vy52 >= 1
 vx1 + vy1 - vx51 - vy51 >= 1
 vx1 + vy1 - vx52 - vy52 >= 1
 vx9 - vx52 >= 1
 vx9 - vx62 >= 1
 vx1 - vx52 >= 1
 vx1 - vx62 >= 1
 vx9 - vy9 - vx62 + vy62 >= 1
 vx9 - vy9 - vx53 + vy53 >= 1
 vx1 - vy1 - vx62 + vy62 >= 1
 vx1 - vy1 - vx53 + vy53 >= 1
 vx9 - vy9 - vx53 + vy53 >= 1
 vx9 - vy9 - vx54 + vy54 >= 1
 vx1 - vy1 - vx53 + vy53 >= 1
 vx1 - vy1 - vx54 + vy54 >= 1
 vx9 - vy9 - vx54 + vy54 >= 1
 vx9 - vy9 - vx55 + vy55 >= 1
 vx1 - vy1 - vx54 + vy54 >= 1
 vx1 - vy1 - vx55 + vy55 >= 1
 vx9 - vy9 - vx55 + vy55 >= 1
 vx9 - vy9 - vx56 + vy56 >= 1
 vx1 - vy1 - vx55 + vy55 >= 1
 vx1 - vy1 - vx56 + vy56 >= 1
 vx9 - vy9 - vx56 + vy56 >= 1
 vx9 - vy9 - vx57 + vy57 >= 1
 vx1 - vy1 - vx56 + vy56 >= 1
 vx1 - vy1 - vx57 + vy57 >= 1
 vx9 - vy9 - vx57 + vy57 >= 1
 vx9 - vy9 - vx58 + vy58 >= 1
 vx1 - vy1 - vx57 + vy57 >= 1
 vx1 - vy1 - vx58 + vy58 >= 1
 vx9 - vy9 - vx58 + vy58 >= 1
 vx9 - vy9 - vx59 + vy59 >= 1
 vx1 - vy1 - vx58 + vy58 >= 1
 vx1 - vy1 - vx59 + vy59 >= 1
 vx9 - vy9 - vx59 + vy59 >= 1
 vx9 - vy9 - vx60 + vy60 >= 1
 vx1 - vy1 - vx59 + vy59 >= 1
 vx1 - vy1 - vx60 + vy60 >= 1
 vx9 - vy9 - vx60 + vy60 >= 1
 vx9 - vy9 - vx61 + vy61 >= 1
 vx1 - vy1 - vx60 + vy60 >= 1
 vx1 - vy1 - vx61 + vy61 >= 1
 q12 - oa12 - ob12 - oc12 - od12 = 0
 a12 + a13 - 2 ua12 - oa12 = 0
 b12 + b13 - 2 ub12 - ob12 = 0
 c12 + c13 - 2 uc12 - oc12 = 0
 d12 + d13 - 2 ud12 - od12 = 0
 vx1 - vy1 - vx6 + vy6 >= 1
 vx1 - vy1 - vx8 + vy8 >= 1
 vx7 - vy7 - vx6 + vy6 >= 1
 vx7 - vy7 - vx8 + vy8 >= 1
 vx1 - vy1 - vx8 + vy8 >= 1
 vx1 - vy1 - vx11 + vy11 >= 1
 vx7 - vy7 - vx8 + vy8 >= 1
 vx7 - vy7 - vx11 + vy11 >= 1
 vx1 - vy1 - vx11 + vy11 >= 1
 vx1 - vy1 - vx13 + vy13 >= 1
 vx7 - vy7 - vx11 + vy11 >= 1
 vx7 - vy7 - vx13 + vy13 >= 1
 vx1 - vy1 - vx13 + vy13 >= 1
 vx1 - vy1 - vx4 + vy4 >= 1
 vx7 - vy7 - vx13 + vy13 >= 1
 vx7 - vy7 - vx4 + vy4 >= 1
 vx1 - vy1 - vx4 + vy4 >= 1
 vx1 - vy1 - vx36 + vy36 >= 1
 vx7 - vy7 - vx4 + vy4 >= 1
 vx7 - vy7 - vx36 + vy36 >= 1
 vx1 - vy1 - vx36 + vy36 >= 1
 vx1 - vy1 - vx39 + vy39 >= 1
 vx7 - vy7 - vx36 + vy36 >= 1
 vx7 - vy7 - vx39 + vy39 >= 1
 vx1 - vy1 - vx39 + vy39 >= 1
 vx1 - vy1 - vx38 + vy38 >= 1
 vx7 - vy7 - vx39 + vy39 >= 1
 vx7 - vy7 - vx38 + vy38 >= 1
 vx1 + vy1 - vx38 - vy38 >= 1
 vx1 + vy1 - vx37 - vy37 >= 1
 vx7 + vy7 - vx38 - vy38 >= 1
 vx7 + vy7 - vx37 - vy37 >= 1
 vx1 - vy1 - vx17 + vy17 <= -1
 vx1 - vy1 - vx27 + vy27 <= -1
 vx7 - vy7 - vx17 + vy17 <= -1
 vx7 - vy7 - vx27 + vy27 <= -1
 vx1 - vy1 - vx27 + vy27 <= -1
 vx1 - vy1 - vx30 + vy30 <= -1
 vx7 - vy7 - vx27 + vy27 <= -1
 vx7 - vy7 - vx30 + vy30 <= -1
 vx1 - vy1 - vx30 + vy30 <= -1
 vx1 - vy1 - vx31 + vy31 <= -1
 vx7 - vy7 - vx30 + vy30 <= -1
 vx7 - vy7 - vx31 + vy31 <= -1
 vx1 - vy1 - vx31 + vy31 <= -1
 vx1 - vy1 - vx33 + vy33 <= -1
 vx7 - vy7 - vx31 + vy31 <= -1
 vx7 - vy7 - vx33 + vy33 <= -1
 vx1 - vy1 - vx33 + vy33 <= -1
 vx1 - vy1 - vx29 + vy29 <= -1
 vx7 - vy7 - vx33 + vy33 <= -1
 vx7 - vy7 - vx29 + vy29 <= -1
 vx1 - vy1 - vx29 + vy29 <= -1
 vx1 - vy1 - vx34 + vy34 <= -1
 vx7 - vy7 - vx29 + vy29 <= -1
 vx7 - vy7 - vx34 + vy34 <= -1
 vx1 - vy1 - vx34 + vy34 <= -1
 vx1 - vy1 - vx32 + vy32 <= -1
 vx7 - vy7 - vx34 + vy34 <= -1
 vx7 - vy7 - vx32 + vy32 <= -1
 vx1 - vy1 - vx32 + vy32 <= -1
 vx1 - vy1 - vx65 + vy65 <= -1
 vx7 - vy7 - vx32 + vy32 <= -1
 vx7 - vy7 - vx65 + vy65 <= -1
 vx1 - vy1 - vx65 + vy65 <= -1
 vx1 - vy1 - vx28 + vy28 <= -1
 vx7 - vy7 - vx65 + vy65 <= -1
 vx7 - vy7 - vx28 + vy28 <= -1
 vx1 - vy1 - vx28 + vy28 <= -1
 vx1 - vy1 - vx18 + vy18 <= -1
 vx7 - vy7 - vx28 + vy28 <= -1
 vx7 - vy7 - vx18 + vy18 <= -1
 vy1 - vy18 >= 1
 vy1 - vy19 >= 1
 vy7 - vy18 >= 1
 vy7 - vy19 >= 1
 vx1 - vy1 - vx19 + vy19 <= -1
 vx1 - vy1 - vx26 + vy26 <= -1
 vx7 - vy7 - vx19 + vy19 <= -1
 vx7 - vy7 - vx26 + vy26 <= -1
 vx1 - vx26 <= -1
 vx1 - vx21 <= -1
 vx7 - vx26 <= -1
 vx7 - vx21 <= -1
 vx1 + vy1 - vx21 - vy21 <= -1
 vx1 + vy1 - vx24 - vy24 <= -1
 vx7 + vy7 - vx21 - vy21 <= -1
 vx7 + vy7 - vx24 - vy24 <= -1
 vx1 + vy1 - vx24 - vy24 <= -1
 vx1 + vy1 - vx23 - vy23 <= -1
 vx7 + vy7 - vx24 - vy24 <= -1
 vx7 + vy7 - vx23 - vy23 <= -1
 vx1 + vy1 - vx23 - vy23 <= -1
 vx1 + vy1 - vx20 - vy20 <= -1
 vx7 + vy7 - vx23 - vy23 <= -1
 vx7 + vy7 - vx20 - vy20 <= -1
 vx1 + vy1 - vx20 - vy20 <= -1
 vx1 + vy1 - vx25 - vy25 <= -1
 vx7 + vy7 - vx20 - vy20 <= -1
 vx7 + vy7 - vx25 - vy25 <= -1
 vx1 + vy1 - vx25 - vy25 <= -1
 vx1 + vy1 - vx22 - vy22 <= -1
 vx7 + vy7 - vx25 - vy25 <= -1
 vx7 + vy7 - vx22 - vy22 <= -1
 vx1 - vy1 - vx41 + vy41 <= -1
 vx1 - vy1 - vx64 + vy64 <= -1
 vx7 - vy7 - vx41 + vy41 <= -1
 vx7 - vy7 - vx64 + vy64 <= -1
 vx1 - vy1 - vx64 + vy64 <= -1
 vx1 - vy1 - vx43 + vy43 <= -1
 vx7 - vy7 - vx64 + vy64 <= -1
 vx7 - vy7 - vx43 + vy43 <= -1
 vx1 - vy1 - vx43 + vy43 <= -1
 vx1 - vy1 - vx44 + vy44 <= -1
 vx7 - vy7 - vx43 + vy43 <= -1
 vx7 - vy7 - vx44 + vy44 <= -1
 vx1 - vy1 - vx44 + vy44 <= -1
 vx1 - vy1 - vx3 + vy3 <= -1
 vx7 - vy7 - vx44 + vy44 <= -1
 vx7 - vy7 - vx3 + vy3 <= -1
 vx1 - vy1 - vx3 + vy3 <= -1
 vx1 - vy1 - vx45 + vy45 <= -1
 vx7 - vy7 - vx3 + vy3 <= -1
 vx7 - vy7 - vx45 + vy45 <= -1
 vx1 - vy1 - vx45 + vy45 <= -1
 vx1 - vy1 - vx46 + vy46 <= -1
 vx7 - vy7 - vx45 + vy45 <= -1
 vx7 - vy7 - vx46 + vy46 <= -1
 vx1 - vy1 - vx46 + vy46 <= -1
 vx1 - vy1 - vx47 + vy47 <= -1
 vx7 - vy7 - vx46 + vy46 <= -1
 vx7 - vy7 - vx47 + vy47 <= -1
 vx1 - vy1 - vx47 + vy47 <= -1
 vx1 - vy1 - vx48 + vy48 <= -1
 vx7 - vy7 - vx47 + vy47 <= -1
 vx7 - vy7 - vx48 + vy48 <= -1
 vx1 - vy1 - vx48 + vy48 <= -1
 vx1 - vy1 - vx35 + vy35 <= -1
 vx7 - vy7 - vx48 + vy48 <= -1
 vx7 - vy7 - vx35 + vy35 <= -1
 vy1 - vy35 >= 1
 vy1 - vy49 >= 1
 vy7 - vy35 >= 1
 vy7 - vy49 >= 1
 vx1 + vy1 - vx49 - vy49 >= 1
 vx1 + vy1 - vx50 - vy50 >= 1
 vx7 + vy7 - vx49 - vy49 >= 1
 vx7 + vy7 - vx50 - vy50 >= 1
 vx1 + vy1 - vx50 - vy50 >= 1
 vx1 + vy1 - vx51 - vy51 >= 1
 vx7 + vy7 - vx50 - vy50 >= 1
 vx7 + vy7 - vx51 - vy51 >= 1
 vx1 + vy1 - vx51 - vy51 >= 1
 vx1 + vy1 - vx52 - vy52 >= 1
 vx7 + vy7 - vx51 - vy51 >= 1
 vx7 + vy7 - vx52 - vy52 >= 1
 vx1 - vx52 >= 1
 vx1 - vx62 >= 1
 vx7 - vx52 >= 1
 vx7 - vx62 >= 1
 vx1 - vy1 - vx62 + vy62 >= 1
 vx1 - vy1 - vx53 + vy53 >= 1
 vx7 - vy7 - vx62 + vy62 >= 1
 vx7 - vy7 - vx53 + vy53 >= 1
 vx1 - vy1 - vx53 + vy53 >= 1
 vx1 - vy1 - vx54 + vy54 >= 1
 vx7 - vy7 - vx53 + vy53 >= 1
 vx7 - vy7 - vx54 + vy54 >= 1
 vx1 - vy1 - vx54 + vy54 >= 1
 vx1 - vy1 - vx55 + vy55 >= 1
 vx7 - vy7 - vx54 + vy54 >= 1
 vx7 - vy7 - vx55 + vy55 >= 1
 vx1 - vy1 - vx55 + vy55 >= 1
 vx1 - vy1 - vx56 + vy56 >= 1
 vx7 - vy7 - vx55 + vy55 >= 1
 vx7 - vy7 - vx56 + vy56 >= 1
 vx1 - vy1 - vx56 + vy56 >= 1
 vx1 - vy1 - vx57 + vy57 >= 1
 vx7 - vy7 - vx56 + vy56 >= 1
 vx7 - vy7 - vx57 + vy57 >= 1
 vx1 - vy1 - vx57 + vy57 >= 1
 vx1 - vy1 - vx58 + vy58 >= 1
 vx7 - vy7 - vx57 + vy57 >= 1
 vx7 - vy7 - vx58 + vy58 >= 1
 vx1 - vy1 - vx58 + vy58 >= 1
 vx1 - vy1 - vx59 + vy59 >= 1
 vx7 - vy7 - vx58 + vy58 >= 1
 vx7 - vy7 - vx59 + vy59 >= 1
 vx1 - vy1 - vx59 + vy59 >= 1
 vx1 - vy1 - vx60 + vy60 >= 1
 vx7 - vy7 - vx59 + vy59 >= 1
 vx7 - vy7 - vx60 + vy60 >= 1
 vx1 - vy1 - vx60 + vy60 >= 1
 vx1 - vy1 - vx61 + vy61 >= 1
 vx7 - vy7 - vx60 + vy60 >= 1
 vx7 - vy7 - vx61 + vy61 >= 1
 q13 - oa13 - ob13 - oc13 - od13 = 0
 a13 + a14 - 2 ua13 - oa13 = 0
 b13 + b14 - 2 ub13 - ob13 = 0
 c13 + c14 - 2 uc13 - oc13 = 0
 d13 + d14 - 2 ud13 - od13 = 0
 vx7 - vy7 - vx8 + vy8 >= 1
 vx7 - vy7 - vx11 + vy11 >= 1
 vx6 - vy6 - vx8 + vy8 >= 1
 vx6 - vy6 - vx11 + vy11 >= 1
 vx7 - vy7 - vx11 + vy11 >= 1
 vx7 - vy7 - vx13 + vy13 >= 1
 vx6 - vy6 - vx11 + vy11 >= 1
 vx6 - vy6 - vx13 + vy13 >= 1
 vx7 - vy7 - vx13 + vy13 >= 1
 vx7 - vy7 - vx4 + vy4 >= 1
 vx6 - vy6 - vx13 + vy13 >= 1
 vx6 - vy6 - vx4 + vy4 >= 1
 vx7 - vy7 - vx4 + vy4 >= 1
 vx7 - vy7 - vx36 + vy36 >= 1
 vx6 - vy6 - vx4 + vy4 >= 1
 vx6 - vy6 - vx36 + vy36 >= 1
 vx7 - vy7 - vx36 + vy36 >= 1
 vx7 - vy7 - vx39 + vy39 >= 1
 vx6 - vy6 - vx36 + vy36 >= 1
 vx6 - vy6 - vx39 + vy39 >= 1
 vx7 - vy7 - vx39 + vy39 >= 1
 vx7 - vy7 - vx38 + vy38 >= 1
 vx6 - vy6 - vx39 + vy39 >= 1
 vx6 - vy6 - vx38 + vy38 >= 1
 vx7 + vy7 - vx38 - vy38 >= 1
 vx7 + vy7 - vx37 - vy37 >= 1
 vx6 + vy6 - vx38 - vy38 >= 1
 vx6 + vy6 - vx37 - vy37 >= 1
 vx7 - vy7 - vx17 + vy17 <= -1
 vx7 - vy7 - vx27 + vy27 <= -1
 vx6 - vy6 - vx17 + vy17 <= -1
 vx6 - vy6 - vx27 + vy27 <= -1
 vx7 - vy7 - vx27 + vy27 <= -1
 vx7 - vy7 - vx30 + vy30 <= -1
 vx6 - vy6 - vx27 + vy27 <= -1
 vx6 - vy6 - vx30 + vy30 <= -1
 vx7 - vy7 - vx30 + vy30 <= -1
 vx7 - vy7 - vx31 + vy31 <= -1
 vx6 - vy6 - vx30 + vy30 <= -1
 vx6 - vy6 - vx31 + vy31 <= -1
 vx7 - vy7 - vx31 + vy31 <= -1
 vx7 - vy7 - vx33 + vy33 <= -1
 vx6 - vy6 - vx31 + vy31 <= -1
 vx6 - vy6 - vx33 + vy33 <= -1
 vx7 - vy7 - vx33 + vy33 <= -1
 vx7 - vy7 - vx29 + vy29 <= -1
 vx6 - vy6 - vx33 + vy33 <= -1
 vx6 - vy6 - vx29 + vy29 <= -1
 vx7 - vy7 - vx29 + vy29 <= -1
 vx7 - vy7 - vx34 + vy34 <= -1
 vx6 - vy6 - vx29 + vy29 <= -1
 vx6 - vy6 - vx34 + vy34 <= -1
 vx7 - vy7 - vx34 + vy34 <= -1
 vx7 - vy7 - vx32 + vy32 <= -1
 vx6 - vy6 - vx34 + vy34 <= -1
 vx6 - vy6 - vx32 + vy32 <= -1
 vx7 - vy7 - vx32 + vy32 <= -1
 vx7 - vy7 - vx65 + vy65 <= -1
 vx6 - vy6 - vx32 + vy32 <= -1
 vx6 - vy6 - vx65 + vy65 <= -1
 vx7 - vy7 - vx65 + vy65 <= -1
 vx7 - vy7 - vx28 + vy28 <= -1
 vx6 - vy6 - vx65 + vy65 <= -1
 vx6 - vy6 - vx28 + vy28 <= -1
 vx7 - vy7 - vx28 + vy28 <= -1
 vx7 - vy7 - vx18 + vy18 <= -1
 vx6 - vy6 - vx28 + vy28 <= -1
 vx6 - vy6 - vx18 + vy18 <= -1
 vx7 - vy7 - vx18 + vy18 <= -1
 vx7 - vy7 - vx19 + vy19 <= -1
 vx6 - vy6 - vx18 + vy18 <= -1
 vx6 - vy6 - vx19 + vy19 <= -1
 vx7 - vy7 - vx19 + vy19 <= -1
 vx7 - vy7 - vx26 + vy26 <= -1
 vx6 - vy6 - vx19 + vy19 <= -1
 vx6 - vy6 - vx26 + vy26 <= -1
 vx7 - vx26 <= -1
 vx7 - vx21 <= -1
 vx6 - vx26 <= -1
 vx6 - vx21 <= -1
 vx7 + vy7 - vx21 - vy21 <= -1
 vx7 + vy7 - vx24 - vy24 <= -1
 vx6 + vy6 - vx21 - vy21 <= -1
 vx6 + vy6 - vx24 - vy24 <= -1
 vx7 + vy7 - vx24 - vy24 <= -1
 vx7 + vy7 - vx23 - vy23 <= -1
 vx6 + vy6 - vx24 - vy24 <= -1
 vx6 + vy6 - vx23 - vy23 <= -1
 vx7 + vy7 - vx23 - vy23 <= -1
 vx7 + vy7 - vx20 - vy20 <= -1
 vx6 + vy6 - vx23 - vy23 <= -1
 vx6 + vy6 - vx20 - vy20 <= -1
 vx7 + vy7 - vx20 - vy20 <= -1
 vx7 + vy7 - vx25 - vy25 <= -1
 vx6 + vy6 - vx20 - vy20 <= -1
 vx6 + vy6 - vx25 - vy25 <= -1
 vx7 + vy7 - vx25 - vy25 <= -1
 vx7 + vy7 - vx22 - vy22 <= -1
 vx6 + vy6 - vx25 - vy25 <= -1
 vx6 + vy6 - vx22 - vy22 <= -1
 vx7 - vy7 - vx41 + vy41 <= -1
 vx7 - vy7 - vx64 + vy64 <= -1
 vx6 - vy6 - vx41 + vy41 <= -1
 vx6 - vy6 - vx64 + vy64 <= -1
 vx7 - vy7 - vx64 + vy64 <= -1
 vx7 - vy7 - vx43 + vy43 <= -1
 vx6 - vy6 - vx64 + vy64 <= -1
 vx6 - vy6 - vx43 + vy43 <= -1
 vx7 - vy7 - vx43 + vy43 <= -1
 vx7 - vy7 - vx44 + vy44 <= -1
 vx6 - vy6 - vx43 + vy43 <= -1
 vx6 - vy6 - vx44 + vy44 <= -1
 vx7 - vy7 - vx44 + vy44 <= -1
 vx7 - vy7 - vx3 + vy3 <= -1
 vx6 - vy6 - vx44 + vy44 <= -1
 vx6 - vy6 - vx3 + vy3 <= -1
 vx7 - vy7 - vx3 + vy3 <= -1
 vx7 - vy7 - vx45 + vy45 <= -1
 vx6 - vy6 - vx3 + vy3 <= -1
 vx6 - vy6 - vx45 + vy45 <= -1
 vx7 - vy7 - vx45 + vy45 <= -1
 vx7 - vy7 - vx46 + vy46 <= -1
 vx6 - vy6 - vx45 + vy45 <= -1
 vx6 - vy6 - vx46 + vy46 <= -1
 vx7 - vy7 - vx46 + vy46 <= -1
 vx7 - vy7 - vx47 + vy47 <= -1
 vx6 - vy6 - vx46 + vy46 <= -1
 vx6 - vy6 - vx47 + vy47 <= -1
 vx7 - vy7 - vx47 + vy47 <= -1
 vx7 - vy7 - vx48 + vy48 <= -1
 vx6 - vy6 - vx47 + vy47 <= -1
 vx6 - vy6 - vx48 + vy48 <= -1
 vx7 - vy7 - vx48 + vy48 <= -1
 vx7 - vy7 - vx35 + vy35 <= -1
 vx6 - vy6 - vx48 + vy48 <= -1
 vx6 - vy6 - vx35 + vy35 <= -1
 vy7 - vy35 >= 1
 vy7 - vy49 >= 1
 vy6 - vy35 >= 1
 vy6 - vy49 >= 1
 vy7 - vy49 >= 1
 vy7 - vy50 >= 1
 vy6 - vy49 >= 1
 vy6 - vy50 >= 1
 vx7 + vy7 - vx50 - vy50 >= 1
 vx7 + vy7 - vx51 - vy51 >= 1
 vx6 + vy6 - vx50 - vy50 >= 1
 vx6 + vy6 - vx51 - vy51 >= 1
 vx7 + vy7 - vx51 - vy51 >= 1
 vx7 + vy7 - vx52 - vy52 >= 1
 vx6 + vy6 - vx51 - vy51 >= 1
 vx6 + vy6 - vx52 - vy52 >= 1
 vx7 - vx52 >= 1
 vx7 - vx62 >= 1
 vx6 - vx52 >= 1
 vx6 - vx62 >= 1
 vx7 - vy7 - vx62 + vy62 >= 1
 vx7 - vy7 - vx53 + vy53 >= 1
 vx6 - vy6 - vx62 + vy62 >= 1
 vx6 - vy6 - vx53 + vy53 >= 1
 vx7 - vy7 - vx53 + vy53 >= 1
 vx7 - vy7 - vx54 + vy54 >= 1
 vx6 - vy6 - vx53 + vy53 >= 1
 vx6 - vy6 - vx54 + vy54 >= 1
 vx7 - vy7 - vx54 + vy54 >= 1
 vx7 - vy7 - vx55 + vy55 >= 1
 vx6 - vy6 - vx54 + vy54 >= 1
 vx6 - vy6 - vx55 + vy55 >= 1
 vx7 - vy7 - vx55 + vy55 >= 1
 vx7 - vy7 - vx56 + vy56 >= 1
 vx6 - vy6 - vx55 + vy55 >= 1
 vx6 - vy6 - vx56 + vy56 >= 1
 vx7 - vy7 - vx56 + vy56 >= 1
 vx7 - vy7 - vx57 + vy57 >= 1
 vx6 - vy6 - vx56 + vy56 >= 1
 vx6 - vy6 - vx57 + vy57 >= 1
 vx7 - vy7 - vx57 + vy57 >= 1
 vx7 - vy7 - vx58 + vy58 >= 1
 vx6 - vy6 - vx57 + vy57 >= 1
 vx6 - vy6 - vx58 + vy58 >= 1
 vx7 - vy7 - vx58 + vy58 >= 1
 vx7 - vy7 - vx59 + vy59 >= 1
 vx6 - vy6 - vx58 + vy58 >= 1
 vx6 - vy6 - vx59 + vy59 >= 1
 vx7 - vy7 - vx59 + vy59 >= 1
 vx7 - vy7 - vx60 + vy60 >= 1
 vx6 - vy6 - vx59 + vy59 >= 1
 vx6 - vy6 - vx60 + vy60 >= 1
 vx7 - vy7 - vx60 + vy60 >= 1
 vx7 - vy7 - vx61 + vy61 >= 1
 vx6 - vy6 - vx60 + vy60 >= 1
 vx6 - vy6 - vx61 + vy61 >= 1
 q14 - oa14 - ob14 - oc14 - od14 = 0
 a14 + a15 - 2 ua14 - oa14 = 0
 b14 + b15 - 2 ub14 - ob14 = 0
 c14 + c15 - 2 uc14 - oc14 = 0
 d14 + d15 - 2 ud14 - od14 = 0
 vx6 - vy6 - vx11 + vy11 >= 1
 vx6 - vy6 - vx13 + vy13 >= 1
 vx8 - vy8 - vx11 + vy11 >= 1
 vx8 - vy8 - vx13 + vy13 >= 1
 vx6 - vy6 - vx13 + vy13 >= 1
 vx6 - vy6 - vx4 + vy4 >= 1
 vx8 - vy8 - vx13 + vy13 >= 1
 vx8 - vy8 - vx4 + vy4 >= 1
 vx6 - vy6 - vx4 + vy4 >= 1
 vx6 - vy6 - vx36 + vy36 >= 1
 vx8 - vy8 - vx4 + vy4 >= 1
 vx8 - vy8 - vx36 + vy36 >= 1
 vx6 - vy6 - vx36 + vy36 >= 1
 vx6 - vy6 - vx39 + vy39 >= 1
 vx8 - vy8 - vx36 + vy36 >= 1
 vx8 - vy8 - vx39 + vy39 >= 1
 vx6 + vy6 - vx39 - vy39 >= 1
 vx6 + vy6 - vx38 - vy38 >= 1
 vx8 + vy8 - vx39 - vy39 >= 1
 vx8 + vy8 - vx38 - vy38 >= 1
 vx6 + vy6 - vx38 - vy38 >= 1
 vx6 + vy6 - vx37 - vy37 >= 1
 vx8 + vy8 - vx38 - vy38 >= 1
 vx8 + vy8 - vx37 - vy37 >= 1
 vx6 - vy6 - vx17 + vy17 <= -1
 vx6 - vy6 - vx27 + vy27 <= -1
 vx8 - vy8 - vx17 + vy17 <= -1
 vx8 - vy8 - vx27 + vy27 <= -1
 vx6 - vy6 - vx27 + vy27 <= -1
 vx6 - vy6 - vx30 + vy30 <= -1
 vx8 - vy8 - vx27 + vy27 <= -1
 vx8 - vy8 - vx30 + vy30 <= -1
 vx6 - vy6 - vx30 + vy30 <= -1
 vx6 - vy6 - vx31 + vy31 <= -1
 vx8 - vy8 - vx30 + vy30 <= -1
 vx8 - vy8 - vx31 + vy31 <= -1
 vx6 - vy6 - vx31 + vy31 <= -1
 vx6 - vy6 - vx33 + vy33 <= -1
 vx8 - vy8 - vx31 + vy31 <= -1
 vx8 - vy8 - vx33 + vy33 <= -1
 vx6 - vy6 - vx33 + vy33 <= -1
 vx6 - vy6 - vx29 + vy29 <= -1
 vx8 - vy8 - vx33 + vy33 <= -1
 vx8 - vy8 - vx29 + vy29 <= -1
 vx6 - vy6 - vx29 + vy29 <= -1
 vx6 - vy6 - vx34 + vy34 <= -1
 vx8 - vy8 - vx29 + vy29 <= -1
 vx8 - vy8 - vx34 + vy34 <= -1
 vx6 - vy6 - vx34 + vy34 <= -1
 vx6 - vy6 - vx32 + vy32 <= -1
 vx8 - vy8 - vx34 + vy34 <= -1
 vx8 - vy8 - vx32 + vy32 <= -1
 vx6 - vy6 - vx32 + vy32 <= -1
 vx6 - vy6 - vx65 + vy65 <= -1
 vx8 - vy8 - vx32 + vy32 <= -1
 vx8 - vy8 - vx65 + vy65 <= -1
 vx6 - vy6 - vx65 + vy65 <= -1
 vx6 - vy6 - vx28 + vy28 <= -1
 vx8 - vy8 - vx65 + vy65 <= -1
 vx8 - vy8 - vx28 + vy28 <= -1
 vx6 - vy6 - vx28 + vy28 <= -1
 vx6 - vy6 - vx18 + vy18 <= -1
 vx8 - vy8 - vx28 + vy28 <= -1
 vx8 - vy8 - vx18 + vy18 <= -1
 vx6 - vy6 - vx18 + vy18 <= -1
 vx6 - vy6 - vx19 + vy19 <= -1
 vx8 - vy8 - vx18 + vy18 <= -1
 vx8 - vy8 - vx19 + vy19 <= -1
 vx6 - vy6 - vx19 + vy19 <= -1
 vx6 - vy6 - vx26 + vy26 <= -1
 vx8 - vy8 - vx19 + vy19 <= -1
 vx8 - vy8 - vx26 + vy26 <= -1
 vx6 - vx26 <= -1
 vx6 - vx21 <= -1
 vx8 - vx26 <= -1
 vx8 - vx21 <= -1
 vx6 + vy6 - vx21 - vy21 <= -1
 vx6 + vy6 - vx24 - vy24 <= -1
 vx8 + vy8 - vx21 - vy21 <= -1
 vx8 + vy8 - vx24 - vy24 <= -1
 vx6 + vy6 - vx24 - vy24 <= -1
 vx6 + vy6 - vx23 - vy23 <= -1
 vx8 + vy8 - vx24 - vy24 <= -1
 vx8 + vy8 - vx23 - vy23 <= -1
 vx6 + vy6 - vx23 - vy23 <= -1
 vx6 + vy6 - vx20 - vy20 <= -1
 vx8 + vy8 - vx23 - vy23 <= -1
 vx8 + vy8 - vx20 - vy20 <= -1
 vx6 + vy6 - vx20 - vy20 <= -1
 vx6 + vy6 - vx25 - vy25 <= -1
 vx8 + vy8 - vx20 - vy20 <= -1
 vx8 + vy8 - vx25 - vy25 <= -1
 vx6 + vy6 - vx25 - vy25 <= -1
 vx6 + vy6 - vx22 - vy22 <= -1
 vx8 + vy8 - vx25 - vy25 <= -1
 vx8 + vy8 - vx22 - vy22 <= -1
 vx6 - vy6 - vx41 + vy41 <= -1
 vx6 - vy6 - vx64 + vy64 <= -1
 vx8 - vy8 - vx41 + vy41 <= -1
 vx8 - vy8 - vx64 + vy64 <= -1
 vx6 - vy6 - vx64 + vy64 <= -1
 vx6 - vy6 - vx43 + vy43 <= -1
 vx8 - vy8 - vx64 + vy64 <= -1
 vx8 - vy8 - vx43 + vy43 <= -1
 vx6 - vy6 - vx43 + vy43 <= -1
 vx6 - vy6 - vx44 + vy44 <= -1
 vx8 - vy8 - vx43 + vy43 <= -1
 vx8 - vy8 - vx44 + vy44 <= -1
 vx6 - vy6 - vx44 + vy44 <= -1
 vx6 - vy6 - vx3 + vy3 <= -1
 vx8 - vy8 - vx44 + vy44 <= -1
 vx8 - vy8 - vx3 + vy3 <= -1
 vx6 - vy6 - vx3 + vy3 <= -1
 vx6 - vy6 - vx45 + vy45 <= -1
 vx8 - vy8 - vx3 + vy3 <= -1
 vx8 - vy8 - vx45 + vy45 <= -1
 vx6 - vy6 - vx45 + vy45 <= -1
 vx6 - vy6 - vx46 + vy46 <= -1
 vx8 - vy8 - vx45 + vy45 <= -1
 vx8 - vy8 - vx46 + vy46 <= -1
 vx6 - vy6 - vx46 + vy46 <= -1
 vx6 - vy6 - vx47 + vy47 <= -1
 vx8 - vy8 - vx46 + vy46 <= -1
 vx8 - vy8 - vx47 + vy47 <= -1
 vx6 - vy6 - vx47 + vy47 <= -1
 vx6 - vy6 - vx48 + vy48 <= -1
 vx8 - vy8 - vx47 + vy47 <= -1
 vx8 - vy8 - vx48 + vy48 <= -1
 vx6 - vy6 - vx48 + vy48 <= -1
 vx6 - vy6 - vx35 + vy35 <= -1
 vx8 - vy8 - vx48 + vy48 <= -1
 vx8 - vy8 - vx35 + vy35 <= -1
 vx6 - vy6 - vx35 + vy35 <= -1
 vx6 - vy6 - vx49 + vy49 <= -1
 vx8 - vy8 - vx35 + vy35 <= -1
 vx8 - vy8 - vx49 + vy49 <= -1
 vy6 - vy49 >= 1
 vy6 - vy50 >= 1
 vy8 - vy49 >= 1
 vy8 - vy50 >= 1
 vx6 + vy6 - vx50 - vy50 >= 1
 vx6 + vy6 - vx51 - vy51 >= 1
 vx8 + vy8 - vx50 - vy50 >= 1
 vx8 + vy8 - vx51 - vy51 >= 1
 vx6 + vy6 - vx51 - vy51 >= 1
 vx6 + vy6 - vx52 - vy52 >= 1
 vx8 + vy8 - vx51 - vy51 >= 1
 vx8 + vy8 - vx52 - vy52 >= 1
 vx6 + vy6 - vx52 - vy52 >= 1
 vx6 + vy6 - vx62 - vy62 >= 1
 vx8 + vy8 - vx52 - vy52 >= 1
 vx8 + vy8 - vx62 - vy62 >= 1
 vx6 - vy6 - vx62 + vy62 >= 1
 vx6 - vy6 - vx53 + vy53 >= 1
 vx8 - vy8 - vx62 + vy62 >= 1
 vx8 - vy8 - vx53 + vy53 >= 1
 vy6 - vy53 <= -1
 vy6 - vy54 <= -1
 vy8 - vy53 <= -1
 vy8 - vy54 <= -1
 vy6 - vy54 <= -1
 vy6 - vy55 <= -1
 vy8 - vy54 <= -1
 vy8 - vy55 <= -1
 vy6 - vy55 <= -1
 vy6 - vy56 <= -1
 vy8 - vy55 <= -1
 vy8 - vy56 <= -1
 vy6 - vy56 <= -1
 vy6 - vy57 <= -1
 vy8 - vy56 <= -1
 vy8 - vy57 <= -1
 vy6 - vy57 <= -1
 vy6 - vy58 <= -1
 vy8 - vy57 <= -1
 vy8 - vy58 <= -1
 vx6 - vy6 - vx58 + vy58 >= 1
 vx6 - vy6 - vx59 + vy59 >= 1
 vx8 - vy8 - vx58 + vy58 >= 1
 vx8 - vy8 - vx59 + vy59 >= 1
 vx6 - vy6 - vx59 + vy59 >= 1
 vx6 - vy6 - vx60 + vy60 >= 1
 vx8 - vy8 - vx59 + vy59 >= 1
 vx8 - vy8 - vx60 + vy60 >= 1
 vx6 - vy6 - vx60 + vy60 >= 1
 vx6 - vy6 - vx61 + vy61 >= 1
 vx8 - vy8 - vx60 + vy60 >= 1
 vx8 - vy8 - vx61 + vy61 >= 1
 q15 - oa15 - ob15 - oc15 - od15 = 0
 a15 + a16 - 2 ua15 - oa15 = 0
 b15 + b16 - 2 ub15 - ob15 = 0
 c15 + c16 - 2 uc15 - oc15 = 0
 d15 + d16 - 2 ud15 - od15 = 0
 vx8 - vy8 - vx13 + vy13 >= 1
 vx8 - vy8 - vx4 + vy4 >= 1
 vx11 - vy11 - vx13 + vy13 >= 1
 vx11 - vy11 - vx4 + vy4 >= 1
 vx8 - vy8 - vx4 + vy4 >= 1
 vx8 - vy8 - vx36 + vy36 >= 1
 vx11 - vy11 - vx4 + vy4 >= 1
 vx11 - vy11 - vx36 + vy36 >= 1
 vx8 + vy8 - vx36 - vy36 >= 1
 vx8 + vy8 - vx39 - vy39 >= 1
 vx11 + vy11 - vx36 - vy36 >= 1
 vx11 + vy11 - vx39 - vy39 >= 1
 vx8 + vy8 - vx39 - vy39 >= 1
 vx8 + vy8 - vx38 - vy38 >= 1
 vx11 + vy11 - vx39 - vy39 >= 1
 vx11 + vy11 - vx38 - vy38 >= 1
 vx8 + vy8 - vx38 - vy38 >= 1
 vx8 + vy8 - vx37 - vy37 >= 1
 vx11 + vy11 - vx38 - vy38 >= 1
 vx11 + vy11 - vx37 - vy37 >= 1
 vx8 - vy8 - vx17 + vy17 <= -1
 vx8 - vy8 - vx27 + vy27 <= -1
 vx11 - vy11 - vx17 + vy17 <= -1
 vx11 - vy11 - vx27 + vy27 <= -1
 vx8 - vy8 - vx27 + vy27 <= -1
 vx8 - vy8 - vx30 + vy30 <= -1
 vx11 - vy11 - vx27 + vy27 <= -1
 vx11 - vy11 - vx30 + vy30 <= -1
 vx8 - vy8 - vx30 + vy30 <= -1
 vx8 - vy8 - vx31 + vy31 <= -1
 vx11 - vy11 - vx30 + vy30 <= -1
 vx11 - vy11 - vx31 + vy31 <= -1
 vx8 - vy8 - vx31 + vy31 <= -1
 vx8 - vy8 - vx33 + vy33 <= -1
 vx11 - vy11 - vx31 + vy31 <= -1
 vx11 - vy11 - vx33 + vy33 <= -1
 vx8 - vy8 - vx33 + vy33 <= -1
 vx8 - vy8 - vx29 + vy29 <= -1
 vx11 - vy11 - vx33 + vy33 <= -1
 vx11 - vy11 - vx29 + vy29 <= -1
 vx8 - vy8 - vx29 + vy29 <= -1
 vx8 - vy8 - vx34 + vy34 <= -1
 vx11 - vy11 - vx29 + vy29 <= -1
 vx11 - vy11 - vx34 + vy34 <= -1
 vx8 - vy8 - vx34 + vy34 <= -1
 vx8 - vy8 - vx32 + vy32 <= -1
 vx11 - vy11 - vx34 + vy34 <= -1
 vx11 - vy11 - vx32 + vy32 <= -1
 vx8 - vy8 - vx32 + vy32 <= -1
 vx8 - vy8 - vx65 + vy65 <= -1
 vx11 - vy11 - vx32 + vy32 <= -1
 vx11 - vy11 - vx65 + vy65 <= -1
 vx8 - vy8 - vx65 + vy65 <= -1
 vx8 - vy8 - vx28 + vy28 <= -1
 vx11 - vy11 - vx65 + vy65 <= -1
 vx11 - vy11 - vx28 + vy28 <= -1
 vx8 - vy8 - vx28 + vy28 <= -1
 vx8 - vy8 - vx18 + vy18 <= -1
 vx11 - vy11 - vx28 + vy28 <= -1
 vx11 - vy11 - vx18 + vy18 <= -1
 vx8 - vy8 - vx18 + vy18 <= -1
 vx8 - vy8 - vx19 + vy19 <= -1
 vx11 - vy11 - vx18 + vy18 <= -1
 vx11 - vy11 - vx19 + vy19 <= -1
 vx8 - vy8 - vx19 + vy19 <= -1
 vx8 - vy8 - vx26 + vy26 <= -1
 vx11 - vy11 - vx19 + vy19 <= -1
 vx11 - vy11 - vx26 + vy26 <= -1
 vx8 - vx26 <= -1
 vx8 - vx21 <= -1
 vx11 - vx26 <= -1
 vx11 - vx21 <= -1
 vx8 + vy8 - vx21 - vy21 <= -1
 vx8 + vy8 - vx24 - vy24 <= -1
 vx11 + vy11 - vx21 - vy21 <= -1
 vx11 + vy11 - vx24 - vy24 <= -1
 vx8 + vy8 - vx24 - vy24 <= -1
 vx8 + vy8 - vx23 - vy23 <= -1
 vx11 + vy11 - vx24 - vy24 <= -1
 vx11 + vy11 - vx23 - vy23 <= -1
 vx8 + vy8 - vx23 - vy23 <= -1
 vx8 + vy8 - vx20 - vy20 <= -1
 vx11 + vy11 - vx23 - vy23 <= -1
 vx11 + vy11 - vx20 - vy20 <= -1
 vx8 + vy8 - vx20 - vy20 <= -1
 vx8 + vy8 - vx25 - vy25 <= -1
 vx11 + vy11 - vx20 - vy20 <= -1
 vx11 + vy11 - vx25 - vy25 <= -1
 vx8 + vy8 - vx25 - vy25 <= -1
 vx8 + vy8 - vx22 - vy22 <= -1
 vx11 + vy11 - vx25 - vy25 <= -1
 vx11 + vy11 - vx22 - vy22 <= -1
 vx8 - vy8 - vx41 + vy41 <= -1
 vx8 - vy8 - vx64 + vy64 <= -1
 vx11 - vy11 - vx41 + vy41 <= -1
 vx11 - vy11 - vx64 + vy64 <= -1
 vx8 - vy8 - vx64 + vy64 <= -1
 vx8 - vy8 - vx43 + vy43 <= -1
 vx11 - vy11 - vx64 + vy64 <= -1
 vx11 - vy11 - vx43 + vy43 <= -1
 vx8 - vy8 - vx43 + vy43 <= -1
 vx8 - vy8 - vx44 + vy44 <= -1
 vx11 - vy11 - vx43 + vy43 <= -1
 vx11 - vy11 - vx44 + vy44 <= -1
 vx8 - vy8 - vx44 + vy44 <= -1
 vx8 - vy8 - vx3 + vy3 <= -1
 vx11 - vy11 - vx44 + vy44 <= -1
 vx11 - vy11 - vx3 + vy3 <= -1
 vx8 - vy8 - vx3 + vy3 <= -1
 vx8 - vy8 - vx45 + vy45 <= -1
 vx11 - vy11 - vx3 + vy3 <= -1
 vx11 - vy11 - vx45 + vy45 <= -1
 vx8 - vy8 - vx45 + vy45 <= -1
 vx8 - vy8 - vx46 + vy46 <= -1
 vx11 - vy11 - vx45 + vy45 <= -1
 vx11 - vy11 - vx46 + vy46 <= -1
 vx8 - vy8 - vx46 + vy46 <= -1
 vx8 - vy8 - vx47 + vy47 <= -1
 vx11 - vy11 - vx46 + vy46 <= -1
 vx11 - vy11 - vx47 + vy47 <= -1
 vx8 - vy8 - vx47 + vy47 <= -1
 vx8 - vy8 - vx48 + vy48 <= -1
 vx11 - vy11 - vx47 + vy47 <= -1
 vx11 - vy11 - vx48 + vy48 <= -1
 vx8 - vy8 - vx48 + vy48 <= -1
 vx8 - vy8 - vx35 + vy35 <= -1
 vx11 - vy11 - vx48 + vy48 <= -1
 vx11 - vy11 - vx35 + vy35 <= -1
 vx8 - vy8 - vx35 + vy35 <= -1
 vx8 - vy8 - vx49 + vy49 <= -1
 vx11 - vy11 - vx35 + vy35 <= -1
 vx11 - vy11 - vx49 + vy49 <= -1
 vy8 - vy49 >= 1
 vy8 - vy50 >= 1
 vy11 - vy49 >= 1
 vy11 - vy50 >= 1
 vy8 - vy50 >= 1
 vy8 - vy51 >= 1
 vy11 - vy50 >= 1
 vy11 - vy51 >= 1
 vy8 - vy51 >= 1
 vy8 - vy52 >= 1
 vy11 - vy51 >= 1
 vy11 - vy52 >= 1
 vy8 - vy52 >= 1
 vy8 - vy62 >= 1
 vy11 - vy52 >= 1
 vy11 - vy62 >= 1
 vx8 + vy8 - vx53 - vy53 <= -1
 vx8 + vy8 - vx54 - vy54 <= -1
 vx11 + vy11 - vx53 - vy53 <= -1
 vx11 + vy11 - vx54 - vy54 <= -1
 vx8 + vy8 - vx54 - vy54 <= -1
 vx8 + vy8 - vx55 - vy55 <= -1
 vx11 + vy11 - vx54 - vy54 <= -1
 vx11 + vy11 - vx55 - vy55 <= -1
 vx8 + vy8 - vx55 - vy55 <= -1
 vx8 + vy8 - vx56 - vy56 <= -1
 vx11 + vy11 - vx55 - vy55 <= -1
 vx11 + vy11 - vx56 - vy56 <= -1
 vx8 + vy8 - vx56 - vy56 <= -1
 vx8 + vy8 - vx57 - vy57 <= -1
 vx11 + vy11 - vx56 - vy56 <= -1
 vx11 + vy11 - vx57 - vy57 <= -1
 vx8 + vy8 - vx57 - vy57 <= -1
 vx8 + vy8 - vx58 - vy58 <= -1
 vx11 + vy11 - vx57 - vy57 <= -1
 vx11 + vy11 - vx58 - vy58 <= -1
 vx8 + vy8 - vx58 - vy58 <= -1
 vx8 + vy8 - vx59 - vy59 <= -1
 vx11 + vy11 - vx58 - vy58 <= -1
 vx11 + vy11 - vx59 - vy59 <= -1
 vx8 - vy8 - vx59 + vy59 >= 1
 vx8 - vy8 - vx60 + vy60 >= 1
 vx11 - vy11 - vx59 + vy59 >= 1
 vx11 - vy11 - vx60 + vy60 >= 1
 vx8 - vy8 - vx60 + vy60 >= 1
 vx8 - vy8 - vx61 + vy61 >= 1
 vx11 - vy11 - vx60 + vy60 >= 1
 vx11 - vy11 - vx61 + vy61 >= 1
 q16 - oa16 - ob16 - oc16 - od16 = 0
 a16 + a17 - 2 ua16 - oa16 = 0
 b16 + b17 - 2 ub16 - ob16 = 0
 c16 + c17 - 2 uc16 - oc16 = 0
 d16 + d17 - 2 ud16 - od16 = 0
 vx11 - vy11 - vx4 + vy4 >= 1
 vx11 - vy11 - vx36 + vy36 >= 1
 vx13 - vy13 - vx4 + vy4 >= 1
 vx13 - vy13 - vx36 + vy36 >= 1
 vx11 + vy11 - vx36 - vy36 >= 1
 vx11 + vy11 - vx39 - vy39 >= 1
 vx13 + vy13 - vx36 - vy36 >= 1
 vx13 + vy13 - vx39 - vy39 >= 1
 vx11 + vy11 - vx39 - vy39 >= 1
 vx11 + vy11 - vx38 - vy38 >= 1
 vx13 + vy13 - vx39 - vy39 >= 1
 vx13 + vy13 - vx38 - vy38 >= 1
 vx11 + vy11 - vx38 - vy38 >= 1
 vx11 + vy11 - vx37 - vy37 >= 1
 vx13 + vy13 - vx38 - vy38 >= 1
 vx13 + vy13 - vx37 - vy37 >= 1
 vx11 - vy11 - vx17 + vy17 <= -1
 vx11 - vy11 - vx27 + vy27 <= -1
 vx13 - vy13 - vx17 + vy17 <= -1
 vx13 - vy13 - vx27 + vy27 <= -1
 vx11 - vy11 - vx27 + vy27 <= -1
 vx11 - vy11 - vx30 + vy30 <= -1
 vx13 - vy13 - vx27 + vy27 <= -1
 vx13 - vy13 - vx30 + vy30 <= -1
 vx11 - vy11 - vx30 + vy30 <= -1
 vx11 - vy11 - vx31 + vy31 <= -1
 vx13 - vy13 - vx30 + vy30 <= -1
 vx13 - vy13 - vx31 + vy31 <= -1
 vx11 - vy11 - vx31 + vy31 <= -1
 vx11 - vy11 - vx33 + vy33 <= -1
 vx13 - vy13 - vx31 + vy31 <= -1
 vx13 - vy13 - vx33 + vy33 <= -1
 vx11 - vy11 - vx33 + vy33 <= -1
 vx11 - vy11 - vx29 + vy29 <= -1
 vx13 - vy13 - vx33 + vy33 <= -1
 vx13 - vy13 - vx29 + vy29 <= -1
 vx11 - vy11 - vx29 + vy29 <= -1
 vx11 - vy11 - vx34 + vy34 <= -1
 vx13 - vy13 - vx29 + vy29 <= -1
 vx13 - vy13 - vx34 + vy34 <= -1
 vx11 - vy11 - vx34 + vy34 <= -1
 vx11 - vy11 - vx32 + vy32 <= -1
 vx13 - vy13 - vx34 + vy34 <= -1
 vx13 - vy13 - vx32 + vy32 <= -1
 vx11 - vy11 - vx32 + vy32 <= -1
 vx11 - vy11 - vx65 + vy65 <= -1
 vx13 - vy13 - vx32 + vy32 <= -1
 vx13 - vy13 - vx65 + vy65 <= -1
 vx11 - vy11 - vx65 + vy65 <= -1
 vx11 - vy11 - vx28 + vy28 <= -1
 vx13 - vy13 - vx65 + vy65 <= -1
 vx13 - vy13 - vx28 + vy28 <= -1
 vx11 - vy11 - vx28 + vy28 <= -1
 vx11 - vy11 - vx18 + vy18 <= -1
 vx13 - vy13 - vx28 + vy28 <= -1
 vx13 - vy13 - vx18 + vy18 <= -1
 vx11 - vy11 - vx18 + vy18 <= -1
 vx11 - vy11 - vx19 + vy19 <= -1
 vx13 - vy13 - vx18 + vy18 <= -1
 vx13 - vy13 - vx19 + vy19 <= -1
 vx11 - vy11 - vx19 + vy19 <= -1
 vx11 - vy11 - vx26 + vy26 <= -1
 vx13 - vy13 - vx19 + vy19 <= -1
 vx13 - vy13 - vx26 + vy26 <= -1
 vx11 - vx26 <= -1
 vx11 - vx21 <= -1
 vx13 - vx26 <= -1
 vx13 - vx21 <= -1
 vx11 + vy11 - vx21 - vy21 <= -1
 vx11 + vy11 - vx24 - vy24 <= -1
 vx13 + vy13 - vx21 - vy21 <= -1
 vx13 + vy13 - vx24 - vy24 <= -1
 vx11 + vy11 - vx24 - vy24 <= -1
 vx11 + vy11 - vx23 - vy23 <= -1
 vx13 + vy13 - vx24 - vy24 <= -1
 vx13 + vy13 - vx23 - vy23 <= -1
 vx11 + vy11 - vx23 - vy23 <= -1
 vx11 + vy11 - vx20 - vy20 <= -1
 vx13 + vy13 - vx23 - vy23 <= -1
 vx13 + vy13 - vx20 - vy20 <= -1
 vx11 + vy11 - vx20 - vy20 <= -1
 vx11 + vy11 - vx25 - vy25 <= -1
 vx13 + vy13 - vx20 - vy20 <= -1
 vx13 + vy13 - vx25 - vy25 <= -1
 vx11 + vy11 - vx25 - vy25 <= -1
 vx11 + vy11 - vx22 - vy22 <= -1
 vx13 + vy13 - vx25 - vy25 <= -1
 vx13 + vy13 - vx22 - vy22 <= -1
 vx11 - vy11 - vx41 + vy41 <= -1
 vx11 - vy11 - vx64 + vy64 <= -1
 vx13 - vy13 - vx41 + vy41 <= -1
 vx13 - vy13 - vx64 + vy64 <= -1
 vx11 - vy11 - vx64 + vy64 <= -1
 vx11 - vy11 - vx43 + vy43 <= -1
 vx13 - vy13 - vx64 + vy64 <= -1
 vx13 - vy13 - vx43 + vy43 <= -1
 vx11 - vy11 - vx43 + vy43 <= -1
 vx11 - vy11 - vx44 + vy44 <= -1
 vx13 - vy13 - vx43 + vy43 <= -1
 vx13 - vy13 - vx44 + vy44 <= -1
 vx11 - vy11 - vx44 + vy44 <= -1
 vx11 - vy11 - vx3 + vy3 <= -1
 vx13 - vy13 - vx44 + vy44 <= -1
 vx13 - vy13 - vx3 + vy3 <= -1
 vx11 - vy11 - vx3 + vy3 <= -1
 vx11 - vy11 - vx45 + vy45 <= -1
 vx13 - vy13 - vx3 + vy3 <= -1
 vx13 - vy13 - vx45 + vy45 <= -1
 vx11 - vy11 - vx45 + vy45 <= -1
 vx11 - vy11 - vx46 + vy46 <= -1
 vx13 - vy13 - vx45 + vy45 <= -1
 vx13 - vy13 - vx46 + vy46 <= -1
 vx11 - vy11 - vx46 + vy46 <= -1
 vx11 - vy11 - vx47 + vy47 <= -1
 vx13 - vy13 - vx46 + vy46 <= -1
 vx13 - vy13 - vx47 + vy47 <= -1
 vx11 - vy11 - vx47 + vy47 <= -1
 vx11 - vy11 - vx48 + vy48 <= -1
 vx13 - vy13 - vx47 + vy47 <= -1
 vx13 - vy13 - vx48 + vy48 <= -1
 vx11 - vy11 - vx48 + vy48 <= -1
 vx11 - vy11 - vx35 + vy35 <= -1
 vx13 - vy13 - vx48 + vy48 <= -1
 vx13 - vy13 - vx35 + vy35 <= -1
 vx11 - vy11 - vx35 + vy35 <= -1
 vx11 - vy11 - vx49 + vy49 <= -1
 vx13 - vy13 - vx35 + vy35 <= -1
 vx13 - vy13 - vx49 + vy49 <= -1
 vx11 - vy11 - vx49 + vy49 <= -1
 vx11 - vy11 - vx50 + vy50 <= -1
 vx13 - vy13 - vx49 + vy49 <= -1
 vx13 - vy13 - vx50 + vy50 <= -1
 vx11 - vy11 - vx50 + vy50 <= -1
 vx11 - vy11 - vx51 + vy51 <= -1
 vx13 - vy13 - vx50 + vy50 <= -1
 vx13 - vy13 - vx51 + vy51 <= -1
 vx11 - vy11 - vx51 + vy51 <= -1
 vx11 - vy11 - vx52 + vy52 <= -1
 vx13 - vy13 - vx51 + vy51 <= -1
 vx13 - vy13 - vx52 + vy52 <= -1
 vx11 - vy11 - vx52 + vy52 <= -1
 vx11 - vy11 - vx62 + vy62 <= -1
 vx13 - vy13 - vx52 + vy52 <= -1
 vx13 - vy13 - vx62 + vy62 <= -1
 vx11 - vy11 - vx62 + vy62 <= -1
 vx11 - vy11 - vx53 + vy53 <= -1
 vx13 - vy13 - vx62 + vy62 <= -1
 vx13 - vy13 - vx53 + vy53 <= -1
 vx11 + vy11 - vx53 - vy53 <= -1
 vx11 + vy11 - vx54 - vy54 <= -1
 vx13 + vy13 - vx53 - vy53 <= -1
 vx13 + vy13 - vx54 - vy54 <= -1
 vx11 + vy11 - vx54 - vy54 <= -1
 vx11 + vy11 - vx55 - vy55 <= -1
 vx13 + vy13 - vx54 - vy54 <= -1
 vx13 + vy13 - vx55 - vy55 <= -1
 vx11 + vy11 - vx55 - vy55 <= -1
 vx11 + vy11 - vx56 - vy56 <= -1
 vx13 + vy13 - vx55 - vy55 <= -1
 vx13 + vy13 - vx56 - vy56 <= -1
 vx11 + vy11 - vx56 - vy56 <= -1
 vx11 + vy11 - vx57 - vy57 <= -1
 vx13 + vy13 - vx56 - vy56 <= -1
 vx13 + vy13 - vx57 - vy57 <= -1
 vx11 + vy11 - vx57 - vy57 <= -1
 vx11 + vy11 - vx58 - vy58 <= -1
 vx13 + vy13 - vx57 - vy57 <= -1
 vx13 + vy13 - vx58 - vy58 <= -1
 vx11 + vy11 - vx58 - vy58 <= -1
 vx11 + vy11 - vx59 - vy59 <= -1
 vx13 + vy13 - vx58 - vy58 <= -1
 vx13 + vy13 - vx59 - vy59 <= -1
 vy11 - vy59 <= -1
 vy11 - vy60 <= -1
 vy13 - vy59 <= -1
 vy13 - vy60 <= -1
 vy11 - vy60 <= -1
 vy11 - vy61 <= -1
 vy13 - vy60 <= -1
 vy13 - vy61 <= -1
 q17 - oa17 - ob17 - oc17 - od17 = 0
 a17 + a18 - 2 ua17 - oa17 = 0
 b17 + b18 - 2 ub17 - ob17 = 0
 c17 + c18 - 2 uc17 - oc17 = 0
 d17 + d18 - 2 ud17 - od17 = 0
 vx13 + vy13 - vx36 - vy36 >= 1
 vx13 + vy13 - vx39 - vy39 >= 1
 vx4 + vy4 - vx36 - vy36 >= 1
 vx4 + vy4 - vx39 - vy39 >= 1
 vx13 + vy13 - vx39 - vy39 >= 1
 vx13 + vy13 - vx38 - vy38 >= 1
 vx4 + vy4 - vx39 - vy39 >= 1
 vx4 + vy4 - vx38 - vy38 >= 1
 vx13 + vy13 - vx38 - vy38 >= 1
 vx13 + vy13 - vx37 - vy37 >= 1
 vx4 + vy4 - vx38 - vy38 >= 1
 vx4 + vy4 - vx37 - vy37 >= 1
 vx13 - vy13 - vx17 + vy17 <= -1
 vx13 - vy13 - vx27 + vy27 <= -1
 vx4 - vy4 - vx17 + vy17 <= -1
 vx4 - vy4 - vx27 + vy27 <= -1
 vx13 - vy13 - vx27 + vy27 <= -1
 vx13 - vy13 - vx30 + vy30 <= -1
 vx4 - vy4 - vx27 + vy27 <= -1
 vx4 - vy4 - vx30 + vy30 <= -1
 vx13 - vy13 - vx30 + vy30 <= -1
 vx13 - vy13 - vx31 + vy31 <= -1
 vx4 - vy4 - vx30 + vy30 <= -1
 vx4 - vy4 - vx31 + vy31 <= -1
 vx13 - vy13 - vx31 + vy31 <= -1
 vx13 - vy13 - vx33 + vy33 <= -1
 vx4 - vy4 - vx31 + vy31 <= -1
 vx4 - vy4 - vx33 + vy33 <= -1
 vx13 - vy13 - vx33 + vy33 <= -1
 vx13 - vy13 - vx29 + vy29 <= -1
 vx4 - vy4 - vx33 + vy33 <= -1
 vx4 - vy4 - vx29 + vy29 <= -1
 vx13 - vy13 - vx29 + vy29 <= -1
 vx13 - vy13 - vx34 + vy34 <= -1
 vx4 - vy4 - vx29 + vy29 <= -1
 vx4 - vy4 - vx34 + vy34 <= -1
 vx13 - vy13 - vx34 + vy34 <= -1
 vx13 - vy13 - vx32 + vy32 <= -1
 vx4 - vy4 - vx34 + vy34 <= -1
 vx4 - vy4 - vx32 + vy32 <= -1
 vx13 - vy13 - vx32 + vy32 <= -1
 vx13 - vy13 - vx65 + vy65 <= -1
 vx4 - vy4 - vx32 + vy32 <= -1
 vx4 - vy4 - vx65 + vy65 <= -1
 vx13 - vy13 - vx65 + vy65 <= -1
 vx13 - vy13 - vx28 + vy28 <= -1
 vx4 - vy4 - vx65 + vy65 <= -1
 vx4 - vy4 - vx28 + vy28 <= -1
 vx13 - vy13 - vx28 + vy28 <= -1
 vx13 - vy13 - vx18 + vy18 <= -1
 vx4 - vy4 - vx28 + vy28 <= -1
 vx4 - vy4 - vx18 + vy18 <= -1
 vx13 - vy13 - vx18 + vy18 <= -1
 vx13 - vy13 - vx19 + vy19 <= -1
 vx4 - vy4 - vx18 + vy18 <= -1
 vx4 - vy4 - vx19 + vy19 <= -1
 vx13 - vy13 - vx19 + vy19 <= -1
 vx13 - vy13 - vx26 + vy26 <= -1
 vx4 - vy4 - vx19 + vy19 <= -1
 vx4 - vy4 - vx26 + vy26 <= -1
 vx13 - vy13 - vx26 + vy26 <= -1
 vx13 - vy13 - vx21 + vy21 <= -1
 vx4 - vy4 - vx26 + vy26 <= -1
 vx4 - vy4 - vx21 + vy21 <= -1
 vx13 + vy13 - vx21 - vy21 <= -1
 vx13 + vy13 - vx24 - vy24 <= -1
 vx4 + vy4 - vx21 - vy21 <= -1
 vx4 + vy4 - vx24 - vy24 <= -1
 vx13 + vy13 - vx24 - vy24 <= -1
 vx13 + vy13 - vx23 - vy23 <= -1
 vx4 + vy4 - vx24 - vy24 <= -1
 vx4 + vy4 - vx23 - vy23 <= -1
 vx13 + vy13 - vx23 - vy23 <= -1
 vx13 + vy13 - vx20 - vy20 <= -1
 vx4 + vy4 - vx23 - vy23 <= -1
 vx4 + vy4 - vx20 - vy20 <= -1
 vx13 + vy13 - vx20 - vy20 <= -1
 vx13 + vy13 - vx25 - vy25 <= -1
 vx4 + vy4 - vx20 - vy20 <= -1
 vx4 + vy4 - vx25 - vy25 <= -1
 vx13 + vy13 - vx25 - vy25 <= -1
 vx13 + vy13 - vx22 - vy22 <= -1
 vx4 + vy4 - vx25 - vy25 <= -1
 vx4 + vy4 - vx22 - vy22 <= -1
 vx13 - vy13 - vx41 + vy41 <= -1
 vx13 - vy13 - vx64 + vy64 <= -1
 vx4 - vy4 - vx41 + vy41 <= -1
 vx4 - vy4 - vx64 + vy64 <= -1
 vx13 - vy13 - vx64 + vy64 <= -1
 vx13 - vy13 - vx43 + vy43 <= -1
 vx4 - vy4 - vx64 + vy64 <= -1
 vx4 - vy4 - vx43 + vy43 <= -1
 vx13 - vy13 - vx43 + vy43 <= -1
 vx13 - vy13 - vx44 + vy44 <= -1
 vx4 - vy4 - vx43 + vy43 <= -1
 vx4 - vy4 - vx44 + vy44 <= -1
 vx13 - vy13 - vx44 + vy44 <= -1
 vx13 - vy13 - vx3 + vy3 <= -1
 vx4 - vy4 - vx44 + vy44 <= -1
 vx4 - vy4 - vx3 + vy3 <= -1
 vx13 - vy13 - vx3 + vy3 <= -1
 vx13 - vy13 - vx45 + vy45 <= -1
 vx4 - vy4 - vx3 + vy3 <= -1
 vx4 - vy4 - vx45 + vy45 <= -1
 vx13 - vy13 - vx45 + vy45 <= -1
 vx13 - vy13 - vx46 + vy46 <= -1
 vx4 - vy4 - vx45 + vy45 <= -1
 vx4 - vy4 - vx46 + vy46 <= -1
 vx13 - vy13 - vx46 + vy46 <= -1
 vx13 - vy13 - vx47 + vy47 <= -1
 vx4 - vy4 - vx46 + vy46 <= -1
 vx4 - vy4 - vx47 + vy47 <= -1
 vx13 - vy13 - vx47 + vy47 <= -1
 vx13 - vy13 - vx48 + vy48 <= -1
 vx4 - vy4 - vx47 + vy47 <= -1
 vx4 - vy4 - vx48 + vy48 <= -1
 vx13 - vy13 - vx48 + vy48 <= -1
 vx13 - vy13 - vx35 + vy35 <= -1
 vx4 - vy4 - vx48 + vy48 <= -1
 vx4 - vy4 - vx35 + vy35 <= -1
 vx13 - vy13 - vx35 + vy35 <= -1
 vx13 - vy13 - vx49 + vy49 <= -1
 vx4 - vy4 - vx35 + vy35 <= -1
 vx4 - vy4 - vx49 + vy49 <= -1
 vx13 - vy13 - vx49 + vy49 <= -1
 vx13 - vy13 - vx50 + vy50 <= -1
 vx4 - vy4 - vx49 + vy49 <= -1
 vx4 - vy4 - vx50 + vy50 <= -1
 vx13 - vy13 - vx50 + vy50 <= -1
 vx13 - vy13 - vx51 + vy51 <= -1
 vx4 - vy4 - vx50 + vy50 <= -1
 vx4 - vy4 - vx51 + vy51 <= -1
 vx13 - vy13 - vx51 + vy51 <= -1
 vx13 - vy13 - vx52 + vy52 <= -1
 vx4 - vy4 - vx51 + vy51 <= -1
 vx4 - vy4 - vx52 + vy52 <= -1
 vx13 - vy13 - vx52 + vy52 <= -1
 vx13 - vy13 - vx62 + vy62 <= -1
 vx4 - vy4 - vx52 + vy52 <= -1
 vx4 - vy4 - vx62 + vy62 <= -1
 vx13 - vy13 - vx62 + vy62 <= -1
 vx13 - vy13 - vx53 + vy53 <= -1
 vx4 - vy4 - vx62 + vy62 <= -1
 vx4 - vy4 - vx53 + vy53 <= -1
 vx13 + vy13 - vx53 - vy53 <= -1
 vx13 + vy13 - vx54 - vy54 <= -1
 vx4 + vy4 - vx53 - vy53 <= -1
 vx4 + vy4 - vx54 - vy54 <= -1
 vx13 + vy13 - vx54 - vy54 <= -1
 vx13 + vy13 - vx55 - vy55 <= -1
 vx4 + vy4 - vx54 - vy54 <= -1
 vx4 + vy4 - vx55 - vy55 <= -1
 vx13 + vy13 - vx55 - vy55 <= -1
 vx13 + vy13 - vx56 - vy56 <= -1
 vx4 + vy4 - vx55 - vy55 <= -1
 vx4 + vy4 - vx56 - vy56 <= -1
 vx13 + vy13 - vx56 - vy56 <= -1
 vx13 + vy13 - vx57 - vy57 <= -1
 vx4 + vy4 - vx56 - vy56 <= -1
 vx4 + vy4 - vx57 - vy57 <= -1
 vx13 + vy13 - vx57 - vy57 <= -1
 vx13 + vy13 - vx58 - vy58 <= -1
 vx4 + vy4 - vx57 - vy57 <= -1
 vx4 + vy4 - vx58 - vy58 <= -1
 vx13 + vy13 - vx58 - vy58 <= -1
 vx13 + vy13 - vx59 - vy59 <= -1
 vx4 + vy4 - vx58 - vy58 <= -1
 vx4 + vy4 - vx59 - vy59 <= -1
 vx13 + vy13 - vx59 - vy59 <= -1
 vx13 + vy13 - vx60 - vy60 <= -1
 vx4 + vy4 - vx59 - vy59 <= -1
 vx4 + vy4 - vx60 - vy60 <= -1
 vx13 + vy13 - vx60 - vy60 <= -1
 vx13 + vy13 - vx61 - vy61 <= -1
 vx4 + vy4 - vx60 - vy60 <= -1
 vx4 + vy4 - vx61 - vy61 <= -1
 q18 - oa18 - ob18 - oc18 - od18 = 0
 a18 + a19 - 2 ua18 - oa18 = 0
 b18 + b19 - 2 ub18 - ob18 = 0
 c18 + c19 - 2 uc18 - oc18 = 0
 d18 + d19 - 2 ud18 - od18 = 0
 vx4 + vy4 - vx39 - vy39 >= 1
 vx4 + vy4 - vx38 - vy38 >= 1
 vx36 + vy36 - vx39 - vy39 >= 1
 vx36 + vy36 - vx38 - vy38 >= 1
 vx4 + vy4 - vx38 - vy38 >= 1
 vx4 + vy4 - vx37 - vy37 >= 1
 vx36 + vy36 - vx38 - vy38 >= 1
 vx36 + vy36 - vx37 - vy37 >= 1
 vx4 - vy4 - vx17 + vy17 <= -1
 vx4 - vy4 - vx27 + vy27 <= -1
 vx36 - vy36 - vx17 + vy17 <= -1
 vx36 - vy36 - vx27 + vy27 <= -1
 vx4 - vy4 - vx27 + vy27 <= -1
 vx4 - vy4 - vx30 + vy30 <= -1
 vx36 - vy36 - vx27 + vy27 <= -1
 vx36 - vy36 - vx30 + vy30 <= -1
 vx4 - vy4 - vx30 + vy30 <= -1
 vx4 - vy4 - vx31 + vy31 <= -1
 vx36 - vy36 - vx30 + vy30 <= -1
 vx36 - vy36 - vx31 + vy31 <= -1
 vx4 - vy4 - vx31 + vy31 <= -1
 vx4 - vy4 - vx33 + vy33 <= -1
 vx36 - vy36 - vx31 + vy31 <= -1
 vx36 - vy36 - vx33 + vy33 <= -1
 vx4 - vy4 - vx33 + vy33 <= -1
 vx4 - vy4 - vx29 + vy29 <= -1
 vx36 - vy36 - vx33 + vy33 <= -1
 vx36 - vy36 - vx29 + vy29 <= -1
 vx4 - vy4 - vx29 + vy29 <= -1
 vx4 - vy4 - vx34 + vy34 <= -1
 vx36 - vy36 - vx29 + vy29 <= -1
 vx36 - vy36 - vx34 + vy34 <= -1
 vx4 - vy4 - vx34 + vy34 <= -1
 vx4 - vy4 - vx32 + vy32 <= -1
 vx36 - vy36 - vx34 + vy34 <= -1
 vx36 - vy36 - vx32 + vy32 <= -1
 vx4 - vy4 - vx32 + vy32 <= -1
 vx4 - vy4 - vx65 + vy65 <= -1
 vx36 - vy36 - vx32 + vy32 <= -1
 vx36 - vy36 - vx65 + vy65 <= -1
 vx4 - vy4 - vx65 + vy65 <= -1
 vx4 - vy4 - vx28 + vy28 <= -1
 vx36 - vy36 - vx65 + vy65 <= -1
 vx36 - vy36 - vx28 + vy28 <= -1
 vx4 - vy4 - vx28 + vy28 <= -1
 vx4 - vy4 - vx18 + vy18 <= -1
 vx36 - vy36 - vx28 + vy28 <= -1
 vx36 - vy36 - vx18 + vy18 <= -1
 vx4 - vy4 - vx18 + vy18 <= -1
 vx4 - vy4 - vx19 + vy19 <= -1
 vx36 - vy36 - vx18 + vy18 <= -1
 vx36 - vy36 - vx19 + vy19 <= -1
 vx4 - vy4 - vx19 + vy19 <= -1
 vx4 - vy4 - vx26 + vy26 <= -1
 vx36 - vy36 - vx19 + vy19 <= -1
 vx36 - vy36 - vx26 + vy26 <= -1
 vx4 - vy4 - vx26 + vy26 <= -1
 vx4 - vy4 - vx21 + vy21 <= -1
 vx36 - vy36 - vx26 + vy26 <= -1
 vx36 - vy36 - vx21 + vy21 <= -1
 vx4 - vx21 <= -1
 vx4 - vx24 <= -1
 vx36 - vx21 <= -1
 vx36 - vx24 <= -1
 vx4 + vy4 - vx24 - vy24 <= -1
 vx4 + vy4 - vx23 - vy23 <= -1
 vx36 + vy36 - vx24 - vy24 <= -1
 vx36 + vy36 - vx23 - vy23 <= -1
 vx4 + vy4 - vx23 - vy23 <= -1
 vx4 + vy4 - vx20 - vy20 <= -1
 vx36 + vy36 - vx23 - vy23 <= -1
 vx36 + vy36 - vx20 - vy20 <= -1
 vx4 + vy4 - vx20 - vy20 <= -1
 vx4 + vy4 - vx25 - vy25 <= -1
 vx36 + vy36 - vx20 - vy20 <= -1
 vx36 + vy36 - vx25 - vy25 <= -1
 vx4 + vy4 - vx25 - vy25 <= -1
 vx4 + vy4 - vx22 - vy22 <= -1
 vx36 + vy36 - vx25 - vy25 <= -1
 vx36 + vy36 - vx22 - vy22 <= -1
 vx4 - vy4 - vx41 + vy41 <= -1
 vx4 - vy4 - vx64 + vy64 <= -1
 vx36 - vy36 - vx41 + vy41 <= -1
 vx36 - vy36 - vx64 + vy64 <= -1
 vx4 - vy4 - vx64 + vy64 <= -1
 vx4 - vy4 - vx43 + vy43 <= -1
 vx36 - vy36 - vx64 + vy64 <= -1
 vx36 - vy36 - vx43 + vy43 <= -1
 vx4 - vy4 - vx43 + vy43 <= -1
 vx4 - vy4 - vx44 + vy44 <= -1
 vx36 - vy36 - vx43 + vy43 <= -1
 vx36 - vy36 - vx44 + vy44 <= -1
 vx4 - vy4 - vx44 + vy44 <= -1
 vx4 - vy4 - vx3 + vy3 <= -1
 vx36 - vy36 - vx44 + vy44 <= -1
 vx36 - vy36 - vx3 + vy3 <= -1
 vx4 - vy4 - vx3 + vy3 <= -1
 vx4 - vy4 - vx45 + vy45 <= -1
 vx36 - vy36 - vx3 + vy3 <= -1
 vx36 - vy36 - vx45 + vy45 <= -1
 vx4 - vy4 - vx45 + vy45 <= -1
 vx4 - vy4 - vx46 + vy46 <= -1
 vx36 - vy36 - vx45 + vy45 <= -1
 vx36 - vy36 - vx46 + vy46 <= -1
 vx4 - vy4 - vx46 + vy46 <= -1
 vx4 - vy4 - vx47 + vy47 <= -1
 vx36 - vy36 - vx46 + vy46 <= -1
 vx36 - vy36 - vx47 + vy47 <= -1
 vx4 - vy4 - vx47 + vy47 <= -1
 vx4 - vy4 - vx48 + vy48 <= -1
 vx36 - vy36 - vx47 + vy47 <= -1
 vx36 - vy36 - vx48 + vy48 <= -1
 vx4 - vy4 - vx48 + vy48 <= -1
 vx4 - vy4 - vx35 + vy35 <= -1
 vx36 - vy36 - vx48 + vy48 <= -1
 vx36 - vy36 - vx35 + vy35 <= -1
 vx4 - vy4 - vx35 + vy35 <= -1
 vx4 - vy4 - vx49 + vy49 <= -1
 vx36 - vy36 - vx35 + vy35 <= -1
 vx36 - vy36 - vx49 + vy49 <= -1
 vx4 - vy4 - vx49 + vy49 <= -1
 vx4 - vy4 - vx50 + vy50 <= -1
 vx36 - vy36 - vx49 + vy49 <= -1
 vx36 - vy36 - vx50 + vy50 <= -1
 vx4 - vy4 - vx50 + vy50 <= -1
 vx4 - vy4 - vx51 + vy51 <= -1
 vx36 - vy36 - vx50 + vy50 <= -1
 vx36 - vy36 - vx51 + vy51 <= -1
 vx4 - vy4 - vx51 + vy51 <= -1
 vx4 - vy4 - vx52 + vy52 <= -1
 vx36 - vy36 - vx51 + vy51 <= -1
 vx36 - vy36 - vx52 + vy52 <= -1
 vx4 - vy4 - vx52 + vy52 <= -1
 vx4 - vy4 - vx62 + vy62 <= -1
 vx36 - vy36 - vx52 + vy52 <= -1
 vx36 - vy36 - vx62 + vy62 <= -1
 vx4 - vy4 - vx62 + vy62 <= -1
 vx4 - vy4 - vx53 + vy53 <= -1
 vx36 - vy36 - vx62 + vy62 <= -1
 vx36 - vy36 - vx53 + vy53 <= -1
 vx4 - vx53 <= -1
 vx4 - vx54 <= -1
 vx36 - vx53 <= -1
 vx36 - vx54 <= -1
 vx4 + vy4 - vx54 - vy54 <= -1
 vx4 + vy4 - vx55 - vy55 <= -1
 vx36 + vy36 - vx54 - vy54 <= -1
 vx36 + vy36 - vx55 - vy55 <= -1
 vx4 + vy4 - vx55 - vy55 <= -1
 vx4 + vy4 - vx56 - vy56 <= -1
 vx36 + vy36 - vx55 - vy55 <= -1
 vx36 + vy36 - vx56 - vy56 <= -1
 vx4 + vy4 - vx56 - vy56 <= -1
 vx4 + vy4 - vx57 - vy57 <= -1
 vx36 + vy36 - vx56 - vy56 <= -1
 vx36 + vy36 - vx57 - vy57 <= -1
 vx4 + vy4 - vx57 - vy57 <= -1
 vx4 + vy4 - vx58 - vy58 <= -1
 vx36 + vy36 - vx57 - vy57 <= -1
 vx36 + vy36 - vx58 - vy58 <= -1
 vx4 + vy4 - vx58 - vy58 <= -1
 vx4 + vy4 - vx59 - vy59 <= -1
 vx36 + vy36 - vx58 - vy58 <= -1
 vx36 + vy36 - vx59 - vy59 <= -1
 vx4 + vy4 - vx59 - vy59 <= -1
 vx4 + vy4 - vx60 - vy60 <= -1
 vx36 + vy36 - vx59 - vy59 <= -1
 vx36 + vy36 - vx60 - vy60 <= -1
 vx4 + vy4 - vx60 - vy60 <= -1
 vx4 + vy4 - vx61 - vy61 <= -1
 vx36 + vy36 - vx60 - vy60 <= -1
 vx36 + vy36 - vx61 - vy61 <= -1
 q19 - oa19 - ob19 - oc19 - od19 = 0
 a19 + a20 - 2 ua19 - oa19 = 0
 b19 + b20 - 2 ub19 - ob19 = 0
 c19 + c20 - 2 uc19 - oc19 = 0
 d19 + d20 - 2 ud19 - od19 = 0
 vx36 + vy36 - vx38 - vy38 >= 1
 vx36 + vy36 - vx37 - vy37 >= 1
 vx39 + vy39 - vx38 - vy38 >= 1
 vx39 + vy39 - vx37 - vy37 >= 1
 vx36 - vy36 - vx17 + vy17 <= -1
 vx36 - vy36 - vx27 + vy27 <= -1
 vx39 - vy39 - vx17 + vy17 <= -1
 vx39 - vy39 - vx27 + vy27 <= -1
 vx36 - vy36 - vx27 + vy27 <= -1
 vx36 - vy36 - vx30 + vy30 <= -1
 vx39 - vy39 - vx27 + vy27 <= -1
 vx39 - vy39 - vx30 + vy30 <= -1
 vx36 - vy36 - vx30 + vy30 <= -1
 vx36 - vy36 - vx31 + vy31 <= -1
 vx39 - vy39 - vx30 + vy30 <= -1
 vx39 - vy39 - vx31 + vy31 <= -1
 vx36 - vy36 - vx31 + vy31 <= -1
 vx36 - vy36 - vx33 + vy33 <= -1
 vx39 - vy39 - vx31 + vy31 <= -1
 vx39 - vy39 - vx33 + vy33 <= -1
 vx36 - vy36 - vx33 + vy33 <= -1
 vx36 - vy36 - vx29 + vy29 <= -1
 vx39 - vy39 - vx33 + vy33 <= -1
 vx39 - vy39 - vx29 + vy29 <= -1
 vx36 - vy36 - vx29 + vy29 <= -1
 vx36 - vy36 - vx34 + vy34 <= -1
 vx39 - vy39 - vx29 + vy29 <= -1
 vx39 - vy39 - vx34 + vy34 <= -1
 vx36 - vy36 - vx34 + vy34 <= -1
 vx36 - vy36 - vx32 + vy32 <= -1
 vx39 - vy39 - vx34 + vy34 <= -1
 vx39 - vy39 - vx32 + vy32 <= -1
 vx36 - vy36 - vx32 + vy32 <= -1
 vx36 - vy36 - vx65 + vy65 <= -1
 vx39 - vy39 - vx32 + vy32 <= -1
 vx39 - vy39 - vx65 + vy65 <= -1
 vx36 - vy36 - vx65 + vy65 <= -1
 vx36 - vy36 - vx28 + vy28 <= -1
 vx39 - vy39 - vx65 + vy65 <= -1
 vx39 - vy39 - vx28 + vy28 <= -1
 vx36 - vy36 - vx28 + vy28 <= -1
 vx36 - vy36 - vx18 + vy18 <= -1
 vx39 - vy39 - vx28 + vy28 <= -1
 vx39 - vy39 - vx18 + vy18 <= -1
 vx36 - vy36 - vx18 + vy18 <= -1
 vx36 - vy36 - vx19 + vy19 <= -1
 vx39 - vy39 - vx18 + vy18 <= -1
 vx39 - vy39 - vx19 + vy19 <= -1
 vx36 - vy36 - vx19 + vy19 <= -1
 vx36 - vy36 - vx26 + vy26 <= -1
 vx39 - vy39 - vx19 + vy19 <= -1
 vx39 - vy39 - vx26 + vy26 <= -1
 vx36 - vx26 <= -1
 vx36 - vx21 <= -1
 vx39 - vx26 <= -1
 vx39 - vx21 <= -1
 vx36 + vy36 - vx21 - vy21 <= -1
 vx36 + vy36 - vx24 - vy24 <= -1
 vx39 + vy39 - vx21 - vy21 <= -1
 vx39 + vy39 - vx24 - vy24 <= -1
 vx36 + vy36 - vx24 - vy24 <= -1
 vx36 + vy36 - vx23 - vy23 <= -1
 vx39 + vy39 - vx24 - vy24 <= -1
 vx39 + vy39 - vx23 - vy23 <= -1
 vx36 + vy36 - vx23 - vy23 <= -1
 vx36 + vy36 - vx20 - vy20 <= -1
 vx39 + vy39 - vx23 - vy23 <= -1
 vx39 + vy39 - vx20 - vy20 <= -1
 vx36 + vy36 - vx20 - vy20 <= -1
 vx36 + vy36 - vx25 - vy25 <= -1
 vx39 + vy39 - vx20 - vy20 <= -1
 vx39 + vy39 - vx25 - vy25 <= -1
 vx36 + vy36 - vx25 - vy25 <= -1
 vx36 + vy36 - vx22 - vy22 <= -1
 vx39 + vy39 - vx25 - vy25 <= -1
 vx39 + vy39 - vx22 - vy22 <= -1
 vx36 - vy36 - vx41 + vy41 <= -1
 vx36 - vy36 - vx64 + vy64 <= -1
 vx39 - vy39 - vx41 + vy41 <= -1
 vx39 - vy39 - vx64 + vy64 <= -1
 vx36 - vy36 - vx64 + vy64 <= -1
 vx36 - vy36 - vx43 + vy43 <= -1
 vx39 - vy39 - vx64 + vy64 <= -1
 vx39 - vy39 - vx43 + vy43 <= -1
 vx36 - vy36 - vx43 + vy43 <= -1
 vx36 - vy36 - vx44 + vy44 <= -1
 vx39 - vy39 - vx43 + vy43 <= -1
 vx39 - vy39 - vx44 + vy44 <= -1
 vx36 - vy36 - vx44 + vy44 <= -1
 vx36 - vy36 - vx3 + vy3 <= -1
 vx39 - vy39 - vx44 + vy44 <= -1
 vx39 - vy39 - vx3 + vy3 <= -1
 vx36 - vy36 - vx3 + vy3 <= -1
 vx36 - vy36 - vx45 + vy45 <= -1
 vx39 - vy39 - vx3 + vy3 <= -1
 vx39 - vy39 - vx45 + vy45 <= -1
 vx36 - vy36 - vx45 + vy45 <= -1
 vx36 - vy36 - vx46 + vy46 <= -1
 vx39 - vy39 - vx45 + vy45 <= -1
 vx39 - vy39 - vx46 + vy46 <= -1
 vx36 - vy36 - vx46 + vy46 <= -1
 vx36 - vy36 - vx47 + vy47 <= -1
 vx39 - vy39 - vx46 + vy46 <= -1
 vx39 - vy39 - vx47 + vy47 <= -1
 vx36 - vy36 - vx47 + vy47 <= -1
 vx36 - vy36 - vx48 + vy48 <= -1
 vx39 - vy39 - vx47 + vy47 <= -1
 vx39 - vy39 - vx48 + vy48 <= -1
 vx36 - vy36 - vx48 + vy48 <= -1
 vx36 - vy36 - vx35 + vy35 <= -1
 vx39 - vy39 - vx48 + vy48 <= -1
 vx39 - vy39 - vx35 + vy35 <= -1
 vx36 - vy36 - vx35 + vy35 <= -1
 vx36 - vy36 - vx49 + vy49 <= -1
 vx39 - vy39 - vx35 + vy35 <= -1
 vx39 - vy39 - vx49 + vy49 <= -1
 vx36 - vy36 - vx49 + vy49 <= -1
 vx36 - vy36 - vx50 + vy50 <= -1
 vx39 - vy39 - vx49 + vy49 <= -1
 vx39 - vy39 - vx50 + vy50 <= -1
 vx36 - vy36 - vx50 + vy50 <= -1
 vx36 - vy36 - vx51 + vy51 <= -1
 vx39 - vy39 - vx50 + vy50 <= -1
 vx39 - vy39 - vx51 + vy51 <= -1
 vx36 - vy36 - vx51 + vy51 <= -1
 vx36 - vy36 - vx52 + vy52 <= -1
 vx39 - vy39 - vx51 + vy51 <= -1
 vx39 - vy39 - vx52 + vy52 <= -1
 vx36 - vy36 - vx52 + vy52 <= -1
 vx36 - vy36 - vx62 + vy62 <= -1
 vx39 - vy39 - vx52 + vy52 <= -1
 vx39 - vy39 - vx62 + vy62 <= -1
 vx36 - vy36 - vx62 + vy62 <= -1
 vx36 - vy36 - vx53 + vy53 <= -1
 vx39 - vy39 - vx62 + vy62 <= -1
 vx39 - vy39 - vx53 + vy53 <= -1
 vx36 + vy36 - vx53 - vy53 <= -1
 vx36 + vy36 - vx54 - vy54 <= -1
 vx39 + vy39 - vx53 - vy53 <= -1
 vx39 + vy39 - vx54 - vy54 <= -1
 vx36 + vy36 - vx54 - vy54 <= -1
 vx36 + vy36 - vx55 - vy55 <= -1
 vx39 + vy39 - vx54 - vy54 <= -1
 vx39 + vy39 - vx55 - vy55 <= -1
 vx36 + vy36 - vx55 - vy55 <= -1
 vx36 + vy36 - vx56 - vy56 <= -1
 vx39 + vy39 - vx55 - vy55 <= -1
 vx39 + vy39 - vx56 - vy56 <= -1
 vx36 + vy36 - vx56 - vy56 <= -1
 vx36 + vy36 - vx57 - vy57 <= -1
 vx39 + vy39 - vx56 - vy56 <= -1
 vx39 + vy39 - vx57 - vy57 <= -1
 vx36 + vy36 - vx57 - vy57 <= -1
 vx36 + vy36 - vx58 - vy58 <= -1
 vx39 + vy39 - vx57 - vy57 <= -1
 vx39 + vy39 - vx58 - vy58 <= -1
 vx36 + vy36 - vx58 - vy58 <= -1
 vx36 + vy36 - vx59 - vy59 <= -1
 vx39 + vy39 - vx58 - vy58 <= -1
 vx39 + vy39 - vx59 - vy59 <= -1
 vx36 + vy36 - vx59 - vy59 <= -1
 vx36 + vy36 - vx60 - vy60 <= -1
 vx39 + vy39 - vx59 - vy59 <= -1
 vx39 + vy39 - vx60 - vy60 <= -1
 vx36 + vy36 - vx60 - vy60 <= -1
 vx36 + vy36 - vx61 - vy61 <= -1
 vx39 + vy39 - vx60 - vy60 <= -1
 vx39 + vy39 - vx61 - vy61 <= -1
 q20 - oa20 - ob20 - oc20 - od20 = 0
 a20 + a21 - 2 ua20 - oa20 = 0
 b20 + b21 - 2 ub20 - ob20 = 0
 c20 + c21 - 2 uc20 - oc20 = 0
 d20 + d21 - 2 ud20 - od20 = 0
 vx39 - vy39 - vx17 + vy17 <= -1
 vx39 - vy39 - vx27 + vy27 <= -1
 vx38 - vy38 - vx17 + vy17 <= -1
 vx38 - vy38 - vx27 + vy27 <= -1
 vx39 - vy39 - vx27 + vy27 <= -1
 vx39 - vy39 - vx30 + vy30 <= -1
 vx38 - vy38 - vx27 + vy27 <= -1
 vx38 - vy38 - vx30 + vy30 <= -1
 vx39 - vy39 - vx30 + vy30 <= -1
 vx39 - vy39 - vx31 + vy31 <= -1
 vx38 - vy38 - vx30 + vy30 <= -1
 vx38 - vy38 - vx31 + vy31 <= -1
 vx39 - vy39 - vx31 + vy31 <= -1
 vx39 - vy39 - vx33 + vy33 <= -1
 vx38 - vy38 - vx31 + vy31 <= -1
 vx38 - vy38 - vx33 + vy33 <= -1
 vx39 - vy39 - vx33 + vy33 <= -1
 vx39 - vy39 - vx29 + vy29 <= -1
 vx38 - vy38 - vx33 + vy33 <= -1
 vx38 - vy38 - vx29 + vy29 <= -1
 vx39 - vy39 - vx29 + vy29 <= -1
 vx39 - vy39 - vx34 + vy34 <= -1
 vx38 - vy38 - vx29 + vy29 <= -1
 vx38 - vy38 - vx34 + vy34 <= -1
 vx39 - vy39 - vx34 + vy34 <= -1
 vx39 - vy39 - vx32 + vy32 <= -1
 vx38 - vy38 - vx34 + vy34 <= -1
 vx38 - vy38 - vx32 + vy32 <= -1
 vx39 - vy39 - vx32 + vy32 <= -1
 vx39 - vy39 - vx65 + vy65 <= -1
 vx38 - vy38 - vx32 + vy32 <= -1
 vx38 - vy38 - vx65 + vy65 <= -1
 vx39 - vy39 - vx65 + vy65 <= -1
 vx39 - vy39 - vx28 + vy28 <= -1
 vx38 - vy38 - vx65 + vy65 <= -1
 vx38 - vy38 - vx28 + vy28 <= -1
 vx39 - vy39 - vx28 + vy28 <= -1
 vx39 - vy39 - vx18 + vy18 <= -1
 vx38 - vy38 - vx28 + vy28 <= -1
 vx38 - vy38 - vx18 + vy18 <= -1
 vx39 - vy39 - vx18 + vy18 <= -1
 vx39 - vy39 - vx19 + vy19 <= -1
 vx38 - vy38 - vx18 + vy18 <= -1
 vx38 - vy38 - vx19 + vy19 <= -1
 vx39 - vy39 - vx19 + vy19 <= -1
 vx39 - vy39 - vx26 + vy26 <= -1
 vx38 - vy38 - vx19 + vy19 <= -1
 vx38 - vy38 - vx26 + vy26 <= -1
 vx39 - vx26 <= -1
 vx39 - vx21 <= -1
 vx38 - vx26 <= -1
 vx38 - vx21 <= -1
 vx39 + vy39 - vx21 - vy21 <= -1
 vx39 + vy39 - vx24 - vy24 <= -1
 vx38 + vy38 - vx21 - vy21 <= -1
 vx38 + vy38 - vx24 - vy24 <= -1
 vx39 + vy39 - vx24 - vy24 <= -1
 vx39 + vy39 - vx23 - vy23 <= -1
 vx38 + vy38 - vx24 - vy24 <= -1
 vx38 + vy38 - vx23 - vy23 <= -1
 vx39 + vy39 - vx23 - vy23 <= -1
 vx39 + vy39 - vx20 - vy20 <= -1
 vx38 + vy38 - vx23 - vy23 <= -1
 vx38 + vy38 - vx20 - vy20 <= -1
 vx39 + vy39 - vx20 - vy20 <= -1
 vx39 + vy39 - vx25 - vy25 <= -1
 vx38 + vy38 - vx20 - vy20 <= -1
 vx38 + vy38 - vx25 - vy25 <= -1
 vx39 + vy39 - vx25 - vy25 <= -1
 vx39 + vy39 - vx22 - vy22 <= -1
 vx38 + vy38 - vx25 - vy25 <= -1
 vx38 + vy38 - vx22 - vy22 <= -1
 vx39 - vy39 - vx41 + vy41 <= -1
 vx39 - vy39 - vx64 + vy64 <= -1
 vx38 - vy38 - vx41 + vy41 <= -1
 vx38 - vy38 - vx64 + vy64 <= -1
 vx39 - vy39 - vx64 + vy64 <= -1
 vx39 - vy39 - vx43 + vy43 <= -1
 vx38 - vy38 - vx64 + vy64 <= -1
 vx38 - vy38 - vx43 + vy43 <= -1
 vx39 - vy39 - vx43 + vy43 <= -1
 vx39 - vy39 - vx44 + vy44 <= -1
 vx38 - vy38 - vx43 + vy43 <= -1
 vx38 - vy38 - vx44 + vy44 <= -1
 vx39 - vy39 - vx44 + vy44 <= -1
 vx39 - vy39 - vx3 + vy3 <= -1
 vx38 - vy38 - vx44 + vy44 <= -1
 vx38 - vy38 - vx3 + vy3 <= -1
 vx39 - vy39 - vx3 + vy3 <= -1
 vx39 - vy39 - vx45 + vy45 <= -1
 vx38 - vy38 - vx3 + vy3 <= -1
 vx38 - vy38 - vx45 + vy45 <= -1
 vx39 - vy39 - vx45 + vy45 <= -1
 vx39 - vy39 - vx46 + vy46 <= -1
 vx38 - vy38 - vx45 + vy45 <= -1
 vx38 - vy38 - vx46 + vy46 <= -1
 vx39 - vy39 - vx46 + vy46 <= -1
 vx39 - vy39 - vx47 + vy47 <= -1
 vx38 - vy38 - vx46 + vy46 <= -1
 vx38 - vy38 - vx47 + vy47 <= -1
 vx39 - vy39 - vx47 + vy47 <= -1
 vx39 - vy39 - vx48 + vy48 <= -1
 vx38 - vy38 - vx47 + vy47 <= -1
 vx38 - vy38 - vx48 + vy48 <= -1
 vx39 - vy39 - vx48 + vy48 <= -1
 vx39 - vy39 - vx35 + vy35 <= -1
 vx38 - vy38 - vx48 + vy48 <= -1
 vx38 - vy38 - vx35 + vy35 <= -1
 vx39 - vy39 - vx35 + vy35 <= -1
 vx39 - vy39 - vx49 + vy49 <= -1
 vx38 - vy38 - vx35 + vy35 <= -1
 vx38 - vy38 - vx49 + vy49 <= -1
 vx39 - vy39 - vx49 + vy49 <= -1
 vx39 - vy39 - vx50 + vy50 <= -1
 vx38 - vy38 - vx49 + vy49 <= -1
 vx38 - vy38 - vx50 + vy50 <= -1
 vx39 - vy39 - vx50 + vy50 <= -1
 vx39 - vy39 - vx51 + vy51 <= -1
 vx38 - vy38 - vx50 + vy50 <= -1
 vx38 - vy38 - vx51 + vy51 <= -1
 vx39 - vy39 - vx51 + vy51 <= -1
 vx39 - vy39 - vx52 + vy52 <= -1
 vx38 - vy38 - vx51 + vy51 <= -1
 vx38 - vy38 - vx52 + vy52 <= -1
 vx39 - vy39 - vx52 + vy52 <= -1
 vx39 - vy39 - vx62 + vy62 <= -1
 vx38 - vy38 - vx52 + vy52 <= -1
 vx38 - vy38 - vx62 + vy62 <= -1
 vx39 + vy39 - vx62 - vy62 <= -1
 vx39 + vy39 - vx53 - vy53 <= -1
 vx38 + vy38 - vx62 - vy62 <= -1
 vx38 + vy38 - vx53 - vy53 <= -1
 vx39 + vy39 - vx53 - vy53 <= -1
 vx39 + vy39 - vx54 - vy54 <= -1
 vx38 + vy38 - vx53 - vy53 <= -1
 vx38 + vy38 - vx54 - vy54 <= -1
 vx39 + vy39 - vx54 - vy54 <= -1
 vx39 + vy39 - vx55 - vy55 <= -1
 vx38 + vy38 - vx54 - vy54 <= -1
 vx38 + vy38 - vx55 - vy55 <= -1
 vx39 + vy39 - vx55 - vy55 <= -1
 vx39 + vy39 - vx56 - vy56 <= -1
 vx38 + vy38 - vx55 - vy55 <= -1
 vx38 + vy38 - vx56 - vy56 <= -1
 vx39 + vy39 - vx56 - vy56 <= -1
 vx39 + vy39 - vx57 - vy57 <= -1
 vx38 + vy38 - vx56 - vy56 <= -1
 vx38 + vy38 - vx57 - vy57 <= -1
 vx39 + vy39 - vx57 - vy57 <= -1
 vx39 + vy39 - vx58 - vy58 <= -1
 vx38 + vy38 - vx57 - vy57 <= -1
 vx38 + vy38 - vx58 - vy58 <= -1
 vx39 + vy39 - vx58 - vy58 <= -1
 vx39 + vy39 - vx59 - vy59 <= -1
 vx38 + vy38 - vx58 - vy58 <= -1
 vx38 + vy38 - vx59 - vy59 <= -1
 vx39 + vy39 - vx59 - vy59 <= -1
 vx39 + vy39 - vx60 - vy60 <= -1
 vx38 + vy38 - vx59 - vy59 <= -1
 vx38 + vy38 - vx60 - vy60 <= -1
 vx39 + vy39 - vx60 - vy60 <= -1
 vx39 + vy39 - vx61 - vy61 <= -1
 vx38 + vy38 - vx60 - vy60 <= -1
 vx38 + vy38 - vx61 - vy61 <= -1
 vx38 - vy38 - vx17 + vy17 <= -1
 vx38 - vy38 - vx27 + vy27 <= -1
 vx37 - vy37 - vx17 + vy17 <= -1
 vx37 - vy37 - vx27 + vy27 <= -1
 vx38 - vy38 - vx27 + vy27 <= -1
 vx38 - vy38 - vx30 + vy30 <= -1
 vx37 - vy37 - vx27 + vy27 <= -1
 vx37 - vy37 - vx30 + vy30 <= -1
 vx38 - vy38 - vx30 + vy30 <= -1
 vx38 - vy38 - vx31 + vy31 <= -1
 vx37 - vy37 - vx30 + vy30 <= -1
 vx37 - vy37 - vx31 + vy31 <= -1
 vx38 - vy38 - vx31 + vy31 <= -1
 vx38 - vy38 - vx33 + vy33 <= -1
 vx37 - vy37 - vx31 + vy31 <= -1
 vx37 - vy37 - vx33 + vy33 <= -1
 vx38 - vy38 - vx33 + vy33 <= -1
 vx38 - vy38 - vx29 + vy29 <= -1
 vx37 - vy37 - vx33 + vy33 <= -1
 vx37 - vy37 - vx29 + vy29 <= -1
 vx38 - vy38 - vx29 + vy29 <= -1
 vx38 - vy38 - vx34 + vy34 <= -1
 vx37 - vy37 - vx29 + vy29 <= -1
 vx37 - vy37 - vx34 + vy34 <= -1
 vx38 - vy38 - vx34 + vy34 <= -1
 vx38 - vy38 - vx32 + vy32 <= -1
 vx37 - vy37 - vx34 + vy34 <= -1
 vx37 - vy37 - vx32 + vy32 <= -1
 vx38 - vy38 - vx32 + vy32 <= -1
 vx38 - vy38 - vx65 + vy65 <= -1
 vx37 - vy37 - vx32 + vy32 <= -1
 vx37 - vy37 - vx65 + vy65 <= -1
 vx38 - vy38 - vx65 + vy65 <= -1
 vx38 - vy38 - vx28 + vy28 <= -1
 vx37 - vy37 - vx65 + vy65 <= -1
 vx37 - vy37 - vx28 + vy28 <= -1
 vx38 - vy38 - vx28 + vy28 <= -1
 vx38 - vy38 - vx18 + vy18 <= -1
 vx37 - vy37 - vx28 + vy28 <= -1
 vx37 - vy37 - vx18 + vy18 <= -1
 vx38 - vy38 - vx18 + vy18 <= -1
 vx38 - vy38 - vx19 + vy19 <= -1
 vx37 - vy37 - vx18 + vy18 <= -1
 vx37 - vy37 - vx19 + vy19 <= -1
 vx38 - vy38 - vx19 + vy19 <= -1
 vx38 - vy38 - vx26 + vy26 <= -1
 vx37 - vy37 - vx19 + vy19 <= -1
 vx37 - vy37 - vx26 + vy26 <= -1
 vx38 - vx26 <= -1
 vx38 - vx21 <= -1
 vx37 - vx26 <= -1
 vx37 - vx21 <= -1
 vx38 + vy38 - vx21 - vy21 <= -1
 vx38 + vy38 - vx24 - vy24 <= -1
 vx37 + vy37 - vx21 - vy21 <= -1
 vx37 + vy37 - vx24 - vy24 <= -1
 vx38 + vy38 - vx24 - vy24 <= -1
 vx38 + vy38 - vx23 - vy23 <= -1
 vx37 + vy37 - vx24 - vy24 <= -1
 vx37 + vy37 - vx23 - vy23 <= -1
 vx38 + vy38 - vx23 - vy23 <= -1
 vx38 + vy38 - vx20 - vy20 <= -1
 vx37 + vy37 - vx23 - vy23 <= -1
 vx37 + vy37 - vx20 - vy20 <= -1
 vx38 + vy38 - vx20 - vy20 <= -1
 vx38 + vy38 - vx25 - vy25 <= -1
 vx37 + vy37 - vx20 - vy20 <= -1
 vx37 + vy37 - vx25 - vy25 <= -1
 vx38 + vy38 - vx25 - vy25 <= -1
 vx38 + vy38 - vx22 - vy22 <= -1
 vx37 + vy37 - vx25 - vy25 <= -1
 vx37 + vy37 - vx22 - vy22 <= -1
 vx38 - vy38 - vx41 + vy41 <= -1
 vx38 - vy38 - vx64 + vy64 <= -1
 vx37 - vy37 - vx41 + vy41 <= -1
 vx37 - vy37 - vx64 + vy64 <= -1
 vx38 - vy38 - vx64 + vy64 <= -1
 vx38 - vy38 - vx43 + vy43 <= -1
 vx37 - vy37 - vx64 + vy64 <= -1
 vx37 - vy37 - vx43 + vy43 <= -1
 vx38 - vy38 - vx43 + vy43 <= -1
 vx38 - vy38 - vx44 + vy44 <= -1
 vx37 - vy37 - vx43 + vy43 <= -1
 vx37 - vy37 - vx44 + vy44 <= -1
 vx38 - vy38 - vx44 + vy44 <= -1
 vx38 - vy38 - vx3 + vy3 <= -1
 vx37 - vy37 - vx44 + vy44 <= -1
 vx37 - vy37 - vx3 + vy3 <= -1
 vx38 - vy38 - vx3 + vy3 <= -1
 vx38 - vy38 - vx45 + vy45 <= -1
 vx37 - vy37 - vx3 + vy3 <= -1
 vx37 - vy37 - vx45 + vy45 <= -1
 vx38 - vy38 - vx45 + vy45 <= -1
 vx38 - vy38 - vx46 + vy46 <= -1
 vx37 - vy37 - vx45 + vy45 <= -1
 vx37 - vy37 - vx46 + vy46 <= -1
 vx38 - vy38 - vx46 + vy46 <= -1
 vx38 - vy38 - vx47 + vy47 <= -1
 vx37 - vy37 - vx46 + vy46 <= -1
 vx37 - vy37 - vx47 + vy47 <= -1
 vx38 - vy38 - vx47 + vy47 <= -1
 vx38 - vy38 - vx48 + vy48 <= -1
 vx37 - vy37 - vx47 + vy47 <= -1
 vx37 - vy37 - vx48 + vy48 <= -1
 vx38 - vy38 - vx48 + vy48 <= -1
 vx38 - vy38 - vx35 + vy35 <= -1
 vx37 - vy37 - vx48 + vy48 <= -1
 vx37 - vy37 - vx35 + vy35 <= -1
 vx38 - vy38 - vx35 + vy35 <= -1
 vx38 - vy38 - vx49 + vy49 <= -1
 vx37 - vy37 - vx35 + vy35 <= -1
 vx37 - vy37 - vx49 + vy49 <= -1
 vx38 - vy38 - vx49 + vy49 <= -1
 vx38 - vy38 - vx50 + vy50 <= -1
 vx37 - vy37 - vx49 + vy49 <= -1
 vx37 - vy37 - vx50 + vy50 <= -1
 vx38 - vy38 - vx50 + vy50 <= -1
 vx38 - vy38 - vx51 + vy51 <= -1
 vx37 - vy37 - vx50 + vy50 <= -1
 vx37 - vy37 - vx51 + vy51 <= -1
 vx38 - vy38 - vx51 + vy51 <= -1
 vx38 - vy38 - vx52 + vy52 <= -1
 vx37 - vy37 - vx51 + vy51 <= -1
 vx37 - vy37 - vx52 + vy52 <= -1
 vx38 - vx52 <= -1
 vx38 - vx62 <= -1
 vx37 - vx52 <= -1
 vx37 - vx62 <= -1
 vx38 + vy38 - vx62 - vy62 <= -1
 vx38 + vy38 - vx53 - vy53 <= -1
 vx37 + vy37 - vx62 - vy62 <= -1
 vx37 + vy37 - vx53 - vy53 <= -1
 vx38 + vy38 - vx53 - vy53 <= -1
 vx38 + vy38 - vx54 - vy54 <= -1
 vx37 + vy37 - vx53 - vy53 <= -1
 vx37 + vy37 - vx54 - vy54 <= -1
 vx38 + vy38 - vx54 - vy54 <= -1
 vx38 + vy38 - vx55 - vy55 <= -1
 vx37 + vy37 - vx54 - vy54 <= -1
 vx37 + vy37 - vx55 - vy55 <= -1
 vx38 + vy38 - vx55 - vy55 <= -1
 vx38 + vy38 - vx56 - vy56 <= -1
 vx37 + vy37 - vx55 - vy55 <= -1
 vx37 + vy37 - vx56 - vy56 <= -1
 vx38 + vy38 - vx56 - vy56 <= -1
 vx38 + vy38 - vx57 - vy57 <= -1
 vx37 + vy37 - vx56 - vy56 <= -1
 vx37 + vy37 - vx57 - vy57 <= -1
 vx38 + vy38 - vx57 - vy57 <= -1
 vx38 + vy38 - vx58 - vy58 <= -1
 vx37 + vy37 - vx57 - vy57 <= -1
 vx37 + vy37 - vx58 - vy58 <= -1
 vx38 + vy38 - vx58 - vy58 <= -1
 vx38 + vy38 - vx59 - vy59 <= -1
 vx37 + vy37 - vx58 - vy58 <= -1
 vx37 + vy37 - vx59 - vy59 <= -1
 vx38 + vy38 - vx59 - vy59 <= -1
 vx38 + vy38 - vx60 - vy60 <= -1
 vx37 + vy37 - vx59 - vy59 <= -1
 vx37 + vy37 - vx60 - vy60 <= -1
 vx38 + vy38 - vx60 - vy60 <= -1
 vx38 + vy38 - vx61 - vy61 <= -1
 vx37 + vy37 - vx60 - vy60 <= -1
 vx37 + vy37 - vx61 - vy61 <= -1
 q21 - oa21 - ob21 - oc21 - od21 = 0
 a22 + a23 - 2 ua21 - oa21 = 0
 b22 + b23 - 2 ub21 - ob21 = 0
 c22 + c23 - 2 uc21 - oc21 = 0
 d22 + d23 - 2 ud21 - od21 = 0
 vx17 - vy17 - vx30 + vy30 >= 1
 vx17 - vy17 - vx31 + vy31 >= 1
 vx27 - vy27 - vx30 + vy30 >= 1
 vx27 - vy27 - vx31 + vy31 >= 1
 vx17 - vy17 - vx31 + vy31 >= 1
 vx17 - vy17 - vx33 + vy33 >= 1
 vx27 - vy27 - vx31 + vy31 >= 1
 vx27 - vy27 - vx33 + vy33 >= 1
 vx17 - vy17 - vx33 + vy33 >= 1
 vx17 - vy17 - vx29 + vy29 >= 1
 vx27 - vy27 - vx33 + vy33 >= 1
 vx27 - vy27 - vx29 + vy29 >= 1
 vx17 - vy17 - vx29 + vy29 >= 1
 vx17 - vy17 - vx34 + vy34 >= 1
 vx27 - vy27 - vx29 + vy29 >= 1
 vx27 - vy27 - vx34 + vy34 >= 1
 vx17 - vy17 - vx34 + vy34 >= 1
 vx17 - vy17 - vx32 + vy32 >= 1
 vx27 - vy27 - vx34 + vy34 >= 1
 vx27 - vy27 - vx32 + vy32 >= 1
 vx17 - vy17 - vx32 + vy32 >= 1
 vx17 - vy17 - vx65 + vy65 >= 1
 vx27 - vy27 - vx32 + vy32 >= 1
 vx27 - vy27 - vx65 + vy65 >= 1
 vx17 - vy17 - vx65 + vy65 >= 1
 vx17 - vy17 - vx28 + vy28 >= 1
 vx27 - vy27 - vx65 + vy65 >= 1
 vx27 - vy27 - vx28 + vy28 >= 1
 vx17 - vy17 - vx28 + vy28 >= 1
 vx17 - vy17 - vx18 + vy18 >= 1
 vx27 - vy27 - vx28 + vy28 >= 1
 vx27 - vy27 - vx18 + vy18 >= 1
 vx17 - vy17 - vx18 + vy18 >= 1
 vx17 - vy17 - vx19 + vy19 >= 1
 vx27 - vy27 - vx18 + vy18 >= 1
 vx27 - vy27 - vx19 + vy19 >= 1
 vx17 - vy17 - vx19 + vy19 >= 1
 vx17 - vy17 - vx26 + vy26 >= 1
 vx27 - vy27 - vx19 + vy19 >= 1
 vx27 - vy27 - vx26 + vy26 >= 1
 vx17 - vy17 - vx26 + vy26 >= 1
 vx17 - vy17 - vx21 + vy21 >= 1
 vx27 - vy27 - vx26 + vy26 >= 1
 vx27 - vy27 - vx21 + vy21 >= 1
 vx17 - vy17 - vx21 + vy21 >= 1
 vx17 - vy17 - vx24 + vy24 >= 1
 vx27 - vy27 - vx21 + vy21 >= 1
 vx27 - vy27 - vx24 + vy24 >= 1
 vx17 - vy17 - vx24 + vy24 >= 1
 vx17 - vy17 - vx23 + vy23 >= 1
 vx27 - vy27 - vx24 + vy24 >= 1
 vx27 - vy27 - vx23 + vy23 >= 1
 vx17 - vy17 - vx23 + vy23 >= 1
 vx17 - vy17 - vx20 + vy20 >= 1
 vx27 - vy27 - vx23 + vy23 >= 1
 vx27 - vy27 - vx20 + vy20 >= 1
 vx17 - vy17 - vx20 + vy20 >= 1
 vx17 - vy17 - vx25 + vy25 >= 1
 vx27 - vy27 - vx20 + vy20 >= 1
 vx27 - vy27 - vx25 + vy25 >= 1
 vx17 - vy17 - vx25 + vy25 >= 1
 vx17 - vy17 - vx22 + vy22 >= 1
 vx27 - vy27 - vx25 + vy25 >= 1
 vx27 - vy27 - vx22 + vy22 >= 1
 vx17 + vy17 - vx41 - vy41 <= -1
 vx17 + vy17 - vx64 - vy64 <= -1
 vx27 + vy27 - vx41 - vy41 <= -1
 vx27 + vy27 - vx64 - vy64 <= -1
 vx17 + vy17 - vx64 - vy64 <= -1
 vx17 + vy17 - vx43 - vy43 <= -1
 vx27 + vy27 - vx64 - vy64 <= -1
 vx27 + vy27 - vx43 - vy43 <= -1
 vx17 + vy17 - vx43 - vy43 <= -1
 vx17 + vy17 - vx44 - vy44 <= -1
 vx27 + vy27 - vx43 - vy43 <= -1
 vx27 + vy27 - vx44 - vy44 <= -1
 vx17 + vy17 - vx44 - vy44 <= -1
 vx17 + vy17 - vx3 - vy3 <= -1
 vx27 + vy27 - vx44 - vy44 <= -1
 vx27 + vy27 - vx3 - vy3 <= -1
 vx17 + vy17 - vx3 - vy3 <= -1
 vx17 + vy17 - vx45 - vy45 <= -1
 vx27 + vy27 - vx3 - vy3 <= -1
 vx27 + vy27 - vx45 - vy45 <= -1
 vx17 + vy17 - vx45 - vy45 <= -1
 vx17 + vy17 - vx46 - vy46 <= -1
 vx27 + vy27 - vx45 - vy45 <= -1
 vx27 + vy27 - vx46 - vy46 <= -1
 vx17 + vy17 - vx46 - vy46 <= -1
 vx17 + vy17 - vx47 - vy47 <= -1
 vx27 + vy27 - vx46 - vy46 <= -1
 vx27 + vy27 - vx47 - vy47 <= -1
 vx17 + vy17 - vx47 - vy47 <= -1
 vx17 + vy17 - vx48 - vy48 <= -1
 vx27 + vy27 - vx47 - vy47 <= -1
 vx27 + vy27 - vx48 - vy48 <= -1
 vy17 - vy48 <= -1
 vy17 - vy35 <= -1
 vy27 - vy48 <= -1
 vy27 - vy35 <= -1
 vx17 - vy17 - vx35 + vy35 >= 1
 vx17 - vy17 - vx49 + vy49 >= 1
 vx27 - vy27 - vx35 + vy35 >= 1
 vx27 - vy27 - vx49 + vy49 >= 1
 vx17 - vy17 - vx49 + vy49 >= 1
 vx17 - vy17 - vx50 + vy50 >= 1
 vx27 - vy27 - vx49 + vy49 >= 1
 vx27 - vy27 - vx50 + vy50 >= 1
 vx17 - vy17 - vx50 + vy50 >= 1
 vx17 - vy17 - vx51 + vy51 >= 1
 vx27 - vy27 - vx50 + vy50 >= 1
 vx27 - vy27 - vx51 + vy51 >= 1
 vx17 - vy17 - vx51 + vy51 >= 1
 vx17 - vy17 - vx52 + vy52 >= 1
 vx27 - vy27 - vx51 + vy51 >= 1
 vx27 - vy27 - vx52 + vy52 >= 1
 vx17 - vy17 - vx52 + vy52 >= 1
 vx17 - vy17 - vx62 + vy62 >= 1
 vx27 - vy27 - vx52 + vy52 >= 1
 vx27 - vy27 - vx62 + vy62 >= 1
 vx17 - vy17 - vx62 + vy62 >= 1
 vx17 - vy17 - vx53 + vy53 >= 1
 vx27 - vy27 - vx62 + vy62 >= 1
 vx27 - vy27 - vx53 + vy53 >= 1
 vx17 - vy17 - vx53 + vy53 >= 1
 vx17 - vy17 - vx54 + vy54 >= 1
 vx27 - vy27 - vx53 + vy53 >= 1
 vx27 - vy27 - vx54 + vy54 >= 1
 vx17 - vy17 - vx54 + vy54 >= 1
 vx17 - vy17 - vx55 + vy55 >= 1
 vx27 - vy27 - vx54 + vy54 >= 1
 vx27 - vy27 - vx55 + vy55 >= 1
 vx17 - vy17 - vx55 + vy55 >= 1
 vx17 - vy17 - vx56 + vy56 >= 1
 vx27 - vy27 - vx55 + vy55 >= 1
 vx27 - vy27 - vx56 + vy56 >= 1
 vx17 - vy17 - vx56 + vy56 >= 1
 vx17 - vy17 - vx57 + vy57 >= 1
 vx27 - vy27 - vx56 + vy56 >= 1
 vx27 - vy27 - vx57 + vy57 >= 1
 vx17 - vy17 - vx57 + vy57 >= 1
 vx17 - vy17 - vx58 + vy58 >= 1
 vx27 - vy27 - vx57 + vy57 >= 1
 vx27 - vy27 - vx58 + vy58 >= 1
 vx17 - vy17 - vx58 + vy58 >= 1
 vx17 - vy17 - vx59 + vy59 >= 1
 vx27 - vy27 - vx58 + vy58 >= 1
 vx27 - vy27 - vx59 + vy59 >= 1
 vx17 - vy17 - vx59 + vy59 >= 1
 vx17 - vy17 - vx60 + vy60 >= 1
 vx27 - vy27 - vx59 + vy59 >= 1
 vx27 - vy27 - vx60 + vy60 >= 1
 vx17 - vy17 - vx60 + vy60 >= 1
 vx17 - vy17 - vx61 + vy61 >= 1
 vx27 - vy27 - vx60 + vy60 >= 1
 vx27 - vy27 - vx61 + vy61 >= 1
 q22 - oa22 - ob22 - oc22 - od22 = 0
 a23 + a24 - 2 ua22 - oa22 = 0
 b23 + b24 - 2 ub22 - ob22 = 0
 c23 + c24 - 2 uc22 - oc22 = 0
 d23 + d24 - 2 ud22 - od22 = 0
 vx27 - vy27 - vx31 + vy31 >= 1
 vx27 - vy27 - vx33 + vy33 >= 1
 vx30 - vy30 - vx31 + vy31 >= 1
 vx30 - vy30 - vx33 + vy33 >= 1
 vx27 - vy27 - vx33 + vy33 >= 1
 vx27 - vy27 - vx29 + vy29 >= 1
 vx30 - vy30 - vx33 + vy33 >= 1
 vx30 - vy30 - vx29 + vy29 >= 1
 vx27 - vy27 - vx29 + vy29 >= 1
 vx27 - vy27 - vx34 + vy34 >= 1
 vx30 - vy30 - vx29 + vy29 >= 1
 vx30 - vy30 - vx34 + vy34 >= 1
 vx27 - vy27 - vx34 + vy34 >= 1
 vx27 - vy27 - vx32 + vy32 >= 1
 vx30 - vy30 - vx34 + vy34 >= 1
 vx30 - vy30 - vx32 + vy32 >= 1
 vx27 - vy27 - vx32 + vy32 >= 1
 vx27 - vy27 - vx65 + vy65 >= 1
 vx30 - vy30 - vx32 + vy32 >= 1
 vx30 - vy30 - vx65 + vy65 >= 1
 vx27 - vy27 - vx65 + vy65 >= 1
 vx27 - vy27 - vx28 + vy28 >= 1
 vx30 - vy30 - vx65 + vy65 >= 1
 vx30 - vy30 - vx28 + vy28 >= 1
 vx27 - vy27 - vx28 + vy28 >= 1
 vx27 - vy27 - vx18 + vy18 >= 1
 vx30 - vy30 - vx28 + vy28 >= 1
 vx30 - vy30 - vx18 + vy18 >= 1
 vx27 - vy27 - vx18 + vy18 >= 1
 vx27 - vy27 - vx19 + vy19 >= 1
 vx30 - vy30 - vx18 + vy18 >= 1
 vx30 - vy30 - vx19 + vy19 >= 1
 vx27 - vy27 - vx19 + vy19 >= 1
 vx27 - vy27 - vx26 + vy26 >= 1
 vx30 - vy30 - vx19 + vy19 >= 1
 vx30 - vy30 - vx26 + vy26 >= 1
 vx27 - vy27 - vx26 + vy26 >= 1
 vx27 - vy27 - vx21 + vy21 >= 1
 vx30 - vy30 - vx26 + vy26 >= 1
 vx30 - vy30 - vx21 + vy21 >= 1
 vx27 - vy27 - vx21 + vy21 >= 1
 vx27 - vy27 - vx24 + vy24 >= 1
 vx30 - vy30 - vx21 + vy21 >= 1
 vx30 - vy30 - vx24 + vy24 >= 1
 vx27 - vy27 - vx24 + vy24 >= 1
 vx27 - vy27 - vx23 + vy23 >= 1
 vx30 - vy30 - vx24 + vy24 >= 1
 vx30 - vy30 - vx23 + vy23 >= 1
 vx27 - vy27 - vx23 + vy23 >= 1
 vx27 - vy27 - vx20 + vy20 >= 1
 vx30 - vy30 - vx23 + vy23 >= 1
 vx30 - vy30 - vx20 + vy20 >= 1
 vx27 - vy27 - vx20 + vy20 >= 1
 vx27 - vy27 - vx25 + vy25 >= 1
 vx30 - vy30 - vx20 + vy20 >= 1
 vx30 - vy30 - vx25 + vy25 >= 1
 vx27 - vy27 - vx25 + vy25 >= 1
 vx27 - vy27 - vx22 + vy22 >= 1
 vx30 - vy30 - vx25 + vy25 >= 1
 vx30 - vy30 - vx22 + vy22 >= 1
 vx27 + vy27 - vx41 - vy41 <= -1
 vx27 + vy27 - vx64 - vy64 <= -1
 vx30 + vy30 - vx41 - vy41 <= -1
 vx30 + vy30 - vx64 - vy64 <= -1
 vx27 + vy27 - vx64 - vy64 <= -1
 vx27 + vy27 - vx43 - vy43 <= -1
 vx30 + vy30 - vx64 - vy64 <= -1
 vx30 + vy30 - vx43 - vy43 <= -1
 vx27 + vy27 - vx43 - vy43 <= -1
 vx27 + vy27 - vx44 - vy44 <= -1
 vx30 + vy30 - vx43 - vy43 <= -1
 vx30 + vy30 - vx44 - vy44 <= -1
 vx27 + vy27 - vx44 - vy44 <= -1
 vx27 + vy27 - vx3 - vy3 <= -1
 vx30 + vy30 - vx44 - vy44 <= -1
 vx30 + vy30 - vx3 - vy3 <= -1
 vx27 + vy27 - vx3 - vy3 <= -1
 vx27 + vy27 - vx45 - vy45 <= -1
 vx30 + vy30 - vx3 - vy3 <= -1
 vx30 + vy30 - vx45 - vy45 <= -1
 vx27 + vy27 - vx45 - vy45 <= -1
 vx27 + vy27 - vx46 - vy46 <= -1
 vx30 + vy30 - vx45 - vy45 <= -1
 vx30 + vy30 - vx46 - vy46 <= -1
 vx27 + vy27 - vx46 - vy46 <= -1
 vx27 + vy27 - vx47 - vy47 <= -1
 vx30 + vy30 - vx46 - vy46 <= -1
 vx30 + vy30 - vx47 - vy47 <= -1
 vx27 + vy27 - vx47 - vy47 <= -1
 vx27 + vy27 - vx48 - vy48 <= -1
 vx30 + vy30 - vx47 - vy47 <= -1
 vx30 + vy30 - vx48 - vy48 <= -1
 vy27 - vy48 <= -1
 vy27 - vy35 <= -1
 vy30 - vy48 <= -1
 vy30 - vy35 <= -1
 vx27 - vy27 - vx35 + vy35 >= 1
 vx27 - vy27 - vx49 + vy49 >= 1
 vx30 - vy30 - vx35 + vy35 >= 1
 vx30 - vy30 - vx49 + vy49 >= 1
 vx27 - vy27 - vx49 + vy49 >= 1
 vx27 - vy27 - vx50 + vy50 >= 1
 vx30 - vy30 - vx49 + vy49 >= 1
 vx30 - vy30 - vx50 + vy50 >= 1
 vx27 - vy27 - vx50 + vy50 >= 1
 vx27 - vy27 - vx51 + vy51 >= 1
 vx30 - vy30 - vx50 + vy50 >= 1
 vx30 - vy30 - vx51 + vy51 >= 1
 vx27 - vy27 - vx51 + vy51 >= 1
 vx27 - vy27 - vx52 + vy52 >= 1
 vx30 - vy30 - vx51 + vy51 >= 1
 vx30 - vy30 - vx52 + vy52 >= 1
 vx27 - vy27 - vx52 + vy52 >= 1
 vx27 - vy27 - vx62 + vy62 >= 1
 vx30 - vy30 - vx52 + vy52 >= 1
 vx30 - vy30 - vx62 + vy62 >= 1
 vx27 - vy27 - vx62 + vy62 >= 1
 vx27 - vy27 - vx53 + vy53 >= 1
 vx30 - vy30 - vx62 + vy62 >= 1
 vx30 - vy30 - vx53 + vy53 >= 1
 vx27 - vy27 - vx53 + vy53 >= 1
 vx27 - vy27 - vx54 + vy54 >= 1
 vx30 - vy30 - vx53 + vy53 >= 1
 vx30 - vy30 - vx54 + vy54 >= 1
 vx27 - vy27 - vx54 + vy54 >= 1
 vx27 - vy27 - vx55 + vy55 >= 1
 vx30 - vy30 - vx54 + vy54 >= 1
 vx30 - vy30 - vx55 + vy55 >= 1
 vx27 - vy27 - vx55 + vy55 >= 1
 vx27 - vy27 - vx56 + vy56 >= 1
 vx30 - vy30 - vx55 + vy55 >= 1
 vx30 - vy30 - vx56 + vy56 >= 1
 vx27 - vy27 - vx56 + vy56 >= 1
 vx27 - vy27 - vx57 + vy57 >= 1
 vx30 - vy30 - vx56 + vy56 >= 1
 vx30 - vy30 - vx57 + vy57 >= 1
 vx27 - vy27 - vx57 + vy57 >= 1
 vx27 - vy27 - vx58 + vy58 >= 1
 vx30 - vy30 - vx57 + vy57 >= 1
 vx30 - vy30 - vx58 + vy58 >= 1
 vx27 - vy27 - vx58 + vy58 >= 1
 vx27 - vy27 - vx59 + vy59 >= 1
 vx30 - vy30 - vx58 + vy58 >= 1
 vx30 - vy30 - vx59 + vy59 >= 1
 vx27 - vy27 - vx59 + vy59 >= 1
 vx27 - vy27 - vx60 + vy60 >= 1
 vx30 - vy30 - vx59 + vy59 >= 1
 vx30 - vy30 - vx60 + vy60 >= 1
 vx27 - vy27 - vx60 + vy60 >= 1
 vx27 - vy27 - vx61 + vy61 >= 1
 vx30 - vy30 - vx60 + vy60 >= 1
 vx30 - vy30 - vx61 + vy61 >= 1
 q23 - oa23 - ob23 - oc23 - od23 = 0
 a24 + a25 - 2 ua23 - oa23 = 0
 b24 + b25 - 2 ub23 - ob23 = 0
 c24 + c25 - 2 uc23 - oc23 = 0
 d24 + d25 - 2 ud23 - od23 = 0
 vx30 - vy30 - vx33 + vy33 >= 1
 vx30 - vy30 - vx29 + vy29 >= 1
 vx31 - vy31 - vx33 + vy33 >= 1
 vx31 - vy31 - vx29 + vy29 >= 1
 vx30 - vy30 - vx29 + vy29 >= 1
 vx30 - vy30 - vx34 + vy34 >= 1
 vx31 - vy31 - vx29 + vy29 >= 1
 vx31 - vy31 - vx34 + vy34 >= 1
 vx30 - vy30 - vx34 + vy34 >= 1
 vx30 - vy30 - vx32 + vy32 >= 1
 vx31 - vy31 - vx34 + vy34 >= 1
 vx31 - vy31 - vx32 + vy32 >= 1
 vx30 - vy30 - vx32 + vy32 >= 1
 vx30 - vy30 - vx65 + vy65 >= 1
 vx31 - vy31 - vx32 + vy32 >= 1
 vx31 - vy31 - vx65 + vy65 >= 1
 vx30 - vy30 - vx65 + vy65 >= 1
 vx30 - vy30 - vx28 + vy28 >= 1
 vx31 - vy31 - vx65 + vy65 >= 1
 vx31 - vy31 - vx28 + vy28 >= 1
 vx30 - vy30 - vx28 + vy28 >= 1
 vx30 - vy30 - vx18 + vy18 >= 1
 vx31 - vy31 - vx28 + vy28 >= 1
 vx31 - vy31 - vx18 + vy18 >= 1
 vx30 - vy30 - vx18 + vy18 >= 1
 vx30 - vy30 - vx19 + vy19 >= 1
 vx31 - vy31 - vx18 + vy18 >= 1
 vx31 - vy31 - vx19 + vy19 >= 1
 vx30 - vy30 - vx19 + vy19 >= 1
 vx30 - vy30 - vx26 + vy26 >= 1
 vx31 - vy31 - vx19 + vy19 >= 1
 vx31 - vy31 - vx26 + vy26 >= 1
 vx30 - vy30 - vx26 + vy26 >= 1
 vx30 - vy30 - vx21 + vy21 >= 1
 vx31 - vy31 - vx26 + vy26 >= 1
 vx31 - vy31 - vx21 + vy21 >= 1
 vx30 - vy30 - vx21 + vy21 >= 1
 vx30 - vy30 - vx24 + vy24 >= 1
 vx31 - vy31 - vx21 + vy21 >= 1
 vx31 - vy31 - vx24 + vy24 >= 1
 vx30 - vy30 - vx24 + vy24 >= 1
 vx30 - vy30 - vx23 + vy23 >= 1
 vx31 - vy31 - vx24 + vy24 >= 1
 vx31 - vy31 - vx23 + vy23 >= 1
 vx30 - vy30 - vx23 + vy23 >= 1
 vx30 - vy30 - vx20 + vy20 >= 1
 vx31 - vy31 - vx23 + vy23 >= 1
 vx31 - vy31 - vx20 + vy20 >= 1
 vx30 - vy30 - vx20 + vy20 >= 1
 vx30 - vy30 - vx25 + vy25 >= 1
 vx31 - vy31 - vx20 + vy20 >= 1
 vx31 - vy31 - vx25 + vy25 >= 1
 vx30 + vy30 - vx25 - vy25 <= -1
 vx30 + vy30 - vx22 - vy22 <= -1
 vx31 + vy31 - vx25 - vy25 <= -1
 vx31 + vy31 - vx22 - vy22 <= -1
 vx30 + vy30 - vx41 - vy41 <= -1
 vx30 + vy30 - vx64 - vy64 <= -1
 vx31 + vy31 - vx41 - vy41 <= -1
 vx31 + vy31 - vx64 - vy64 <= -1
 vx30 + vy30 - vx64 - vy64 <= -1
 vx30 + vy30 - vx43 - vy43 <= -1
 vx31 + vy31 - vx64 - vy64 <= -1
 vx31 + vy31 - vx43 - vy43 <= -1
 vx30 + vy30 - vx43 - vy43 <= -1
 vx30 + vy30 - vx44 - vy44 <= -1
 vx31 + vy31 - vx43 - vy43 <= -1
 vx31 + vy31 - vx44 - vy44 <= -1
 vx30 + vy30 - vx44 - vy44 <= -1
 vx30 + vy30 - vx3 - vy3 <= -1
 vx31 + vy31 - vx44 - vy44 <= -1
 vx31 + vy31 - vx3 - vy3 <= -1
 vx30 + vy30 - vx3 - vy3 <= -1
 vx30 + vy30 - vx45 - vy45 <= -1
 vx31 + vy31 - vx3 - vy3 <= -1
 vx31 + vy31 - vx45 - vy45 <= -1
 vx30 + vy30 - vx45 - vy45 <= -1
 vx30 + vy30 - vx46 - vy46 <= -1
 vx31 + vy31 - vx45 - vy45 <= -1
 vx31 + vy31 - vx46 - vy46 <= -1
 vx30 + vy30 - vx46 - vy46 <= -1
 vx30 + vy30 - vx47 - vy47 <= -1
 vx31 + vy31 - vx46 - vy46 <= -1
 vx31 + vy31 - vx47 - vy47 <= -1
 vx30 + vy30 - vx47 - vy47 <= -1
 vx30 + vy30 - vx48 - vy48 <= -1
 vx31 + vy31 - vx47 - vy47 <= -1
 vx31 + vy31 - vx48 - vy48 <= -1
 vy30 - vy48 <= -1
 vy30 - vy35 <= -1
 vy31 - vy48 <= -1
 vy31 - vy35 <= -1
 vx30 - vy30 - vx35 + vy35 >= 1
 vx30 - vy30 - vx49 + vy49 >= 1
 vx31 - vy31 - vx35 + vy35 >= 1
 vx31 - vy31 - vx49 + vy49 >= 1
 vx30 - vy30 - vx49 + vy49 >= 1
 vx30 - vy30 - vx50 + vy50 >= 1
 vx31 - vy31 - vx49 + vy49 >= 1
 vx31 - vy31 - vx50 + vy50 >= 1
 vx30 - vy30 - vx50 + vy50 >= 1
 vx30 - vy30 - vx51 + vy51 >= 1
 vx31 - vy31 - vx50 + vy50 >= 1
 vx31 - vy31 - vx51 + vy51 >= 1
 vx30 - vy30 - vx51 + vy51 >= 1
 vx30 - vy30 - vx52 + vy52 >= 1
 vx31 - vy31 - vx51 + vy51 >= 1
 vx31 - vy31 - vx52 + vy52 >= 1
 vx30 - vy30 - vx52 + vy52 >= 1
 vx30 - vy30 - vx62 + vy62 >= 1
 vx31 - vy31 - vx52 + vy52 >= 1
 vx31 - vy31 - vx62 + vy62 >= 1
 vx30 - vy30 - vx62 + vy62 >= 1
 vx30 - vy30 - vx53 + vy53 >= 1
 vx31 - vy31 - vx62 + vy62 >= 1
 vx31 - vy31 - vx53 + vy53 >= 1
 vx30 - vy30 - vx53 + vy53 >= 1
 vx30 - vy30 - vx54 + vy54 >= 1
 vx31 - vy31 - vx53 + vy53 >= 1
 vx31 - vy31 - vx54 + vy54 >= 1
 vx30 - vy30 - vx54 + vy54 >= 1
 vx30 - vy30 - vx55 + vy55 >= 1
 vx31 - vy31 - vx54 + vy54 >= 1
 vx31 - vy31 - vx55 + vy55 >= 1
 vx30 - vy30 - vx55 + vy55 >= 1
 vx30 - vy30 - vx56 + vy56 >= 1
 vx31 - vy31 - vx55 + vy55 >= 1
 vx31 - vy31 - vx56 + vy56 >= 1
 vx30 - vy30 - vx56 + vy56 >= 1
 vx30 - vy30 - vx57 + vy57 >= 1
 vx31 - vy31 - vx56 + vy56 >= 1
 vx31 - vy31 - vx57 + vy57 >= 1
 vx30 - vy30 - vx57 + vy57 >= 1
 vx30 - vy30 - vx58 + vy58 >= 1
 vx31 - vy31 - vx57 + vy57 >= 1
 vx31 - vy31 - vx58 + vy58 >= 1
 vx30 - vy30 - vx58 + vy58 >= 1
 vx30 - vy30 - vx59 + vy59 >= 1
 vx31 - vy31 - vx58 + vy58 >= 1
 vx31 - vy31 - vx59 + vy59 >= 1
 vx30 - vy30 - vx59 + vy59 >= 1
 vx30 - vy30 - vx60 + vy60 >= 1
 vx31 - vy31 - vx59 + vy59 >= 1
 vx31 - vy31 - vx60 + vy60 >= 1
 vx30 - vy30 - vx60 + vy60 >= 1
 vx30 - vy30 - vx61 + vy61 >= 1
 vx31 - vy31 - vx60 + vy60 >= 1
 vx31 - vy31 - vx61 + vy61 >= 1
 q24 - oa24 - ob24 - oc24 - od24 = 0
 a25 + a26 - 2 ua24 - oa24 = 0
 b25 + b26 - 2 ub24 - ob24 = 0
 c25 + c26 - 2 uc24 - oc24 = 0
 d25 + d26 - 2 ud24 - od24 = 0
 vx31 - vy31 - vx29 + vy29 >= 1
 vx31 - vy31 - vx34 + vy34 >= 1
 vx33 - vy33 - vx29 + vy29 >= 1
 vx33 - vy33 - vx34 + vy34 >= 1
 vx31 + vy31 - vx34 - vy34 <= -1
 vx31 + vy31 - vx32 - vy32 <= -1
 vx33 + vy33 - vx34 - vy34 <= -1
 vx33 + vy33 - vx32 - vy32 <= -1
 vx31 - vy31 - vx32 + vy32 >= 1
 vx31 - vy31 - vx65 + vy65 >= 1
 vx33 - vy33 - vx32 + vy32 >= 1
 vx33 - vy33 - vx65 + vy65 >= 1
 vx31 - vy31 - vx65 + vy65 >= 1
 vx31 - vy31 - vx28 + vy28 >= 1
 vx33 - vy33 - vx65 + vy65 >= 1
 vx33 - vy33 - vx28 + vy28 >= 1
 vx31 - vy31 - vx28 + vy28 >= 1
 vx31 - vy31 - vx18 + vy18 >= 1
 vx33 - vy33 - vx28 + vy28 >= 1
 vx33 - vy33 - vx18 + vy18 >= 1
 vx31 - vy31 - vx18 + vy18 >= 1
 vx31 - vy31 - vx19 + vy19 >= 1
 vx33 - vy33 - vx18 + vy18 >= 1
 vx33 - vy33 - vx19 + vy19 >= 1
 vx31 - vy31 - vx19 + vy19 >= 1
 vx31 - vy31 - vx26 + vy26 >= 1
 vx33 - vy33 - vx19 + vy19 >= 1
 vx33 - vy33 - vx26 + vy26 >= 1
 vx31 - vy31 - vx26 + vy26 >= 1
 vx31 - vy31 - vx21 + vy21 >= 1
 vx33 - vy33 - vx26 + vy26 >= 1
 vx33 - vy33 - vx21 + vy21 >= 1
 vx31 - vy31 - vx21 + vy21 >= 1
 vx31 - vy31 - vx24 + vy24 >= 1
 vx33 - vy33 - vx21 + vy21 >= 1
 vx33 - vy33 - vx24 + vy24 >= 1
 vx31 - vy31 - vx24 + vy24 >= 1
 vx31 - vy31 - vx23 + vy23 >= 1
 vx33 - vy33 - vx24 + vy24 >= 1
 vx33 - vy33 - vx23 + vy23 >= 1
 vx31 - vy31 - vx23 + vy23 >= 1
 vx31 - vy31 - vx20 + vy20 >= 1
 vx33 - vy33 - vx23 + vy23 >= 1
 vx33 - vy33 - vx20 + vy20 >= 1
 vx31 - vy31 - vx20 + vy20 >= 1
 vx31 - vy31 - vx25 + vy25 >= 1
 vx33 - vy33 - vx20 + vy20 >= 1
 vx33 - vy33 - vx25 + vy25 >= 1
 vx31 + vy31 - vx25 - vy25 <= -1
 vx31 + vy31 - vx22 - vy22 <= -1
 vx33 + vy33 - vx25 - vy25 <= -1
 vx33 + vy33 - vx22 - vy22 <= -1
 vx31 + vy31 - vx41 - vy41 <= -1
 vx31 + vy31 - vx64 - vy64 <= -1
 vx33 + vy33 - vx41 - vy41 <= -1
 vx33 + vy33 - vx64 - vy64 <= -1
 vx31 + vy31 - vx64 - vy64 <= -1
 vx31 + vy31 - vx43 - vy43 <= -1
 vx33 + vy33 - vx64 - vy64 <= -1
 vx33 + vy33 - vx43 - vy43 <= -1
 vx31 + vy31 - vx43 - vy43 <= -1
 vx31 + vy31 - vx44 - vy44 <= -1
 vx33 + vy33 - vx43 - vy43 <= -1
 vx33 + vy33 - vx44 - vy44 <= -1
 vx31 + vy31 - vx44 - vy44 <= -1
 vx31 + vy31 - vx3 - vy3 <= -1
 vx33 + vy33 - vx44 - vy44 <= -1
 vx33 + vy33 - vx3 - vy3 <= -1
 vx31 + vy31 - vx3 - vy3 <= -1
 vx31 + vy31 - vx45 - vy45 <= -1
 vx33 + vy33 - vx3 - vy3 <= -1
 vx33 + vy33 - vx45 - vy45 <= -1
 vx31 + vy31 - vx45 - vy45 <= -1
 vx31 + vy31 - vx46 - vy46 <= -1
 vx33 + vy33 - vx45 - vy45 <= -1
 vx33 + vy33 - vx46 - vy46 <= -1
 vx31 + vy31 - vx46 - vy46 <= -1
 vx31 + vy31 - vx47 - vy47 <= -1
 vx33 + vy33 - vx46 - vy46 <= -1
 vx33 + vy33 - vx47 - vy47 <= -1
 vx31 + vy31 - vx47 - vy47 <= -1
 vx31 + vy31 - vx48 - vy48 <= -1
 vx33 + vy33 - vx47 - vy47 <= -1
 vx33 + vy33 - vx48 - vy48 <= -1
 vy31 - vy48 <= -1
 vy31 - vy35 <= -1
 vy33 - vy48 <= -1
 vy33 - vy35 <= -1
 vx31 - vy31 - vx35 + vy35 >= 1
 vx31 - vy31 - vx49 + vy49 >= 1
 vx33 - vy33 - vx35 + vy35 >= 1
 vx33 - vy33 - vx49 + vy49 >= 1
 vx31 - vy31 - vx49 + vy49 >= 1
 vx31 - vy31 - vx50 + vy50 >= 1
 vx33 - vy33 - vx49 + vy49 >= 1
 vx33 - vy33 - vx50 + vy50 >= 1
 vx31 - vy31 - vx50 + vy50 >= 1
 vx31 - vy31 - vx51 + vy51 >= 1
 vx33 - vy33 - vx50 + vy50 >= 1
 vx33 - vy33 - vx51 + vy51 >= 1
 vx31 - vy31 - vx51 + vy51 >= 1
 vx31 - vy31 - vx52 + vy52 >= 1
 vx33 - vy33 - vx51 + vy51 >= 1
 vx33 - vy33 - vx52 + vy52 >= 1
 vx31 - vy31 - vx52 + vy52 >= 1
 vx31 - vy31 - vx62 + vy62 >= 1
 vx33 - vy33 - vx52 + vy52 >= 1
 vx33 - vy33 - vx62 + vy62 >= 1
 vx31 - vy31 - vx62 + vy62 >= 1
 vx31 - vy31 - vx53 + vy53 >= 1
 vx33 - vy33 - vx62 + vy62 >= 1
 vx33 - vy33 - vx53 + vy53 >= 1
 vx31 - vy31 - vx53 + vy53 >= 1
 vx31 - vy31 - vx54 + vy54 >= 1
 vx33 - vy33 - vx53 + vy53 >= 1
 vx33 - vy33 - vx54 + vy54 >= 1
 vx31 - vy31 - vx54 + vy54 >= 1
 vx31 - vy31 - vx55 + vy55 >= 1
 vx33 - vy33 - vx54 + vy54 >= 1
 vx33 - vy33 - vx55 + vy55 >= 1
 vx31 - vy31 - vx55 + vy55 >= 1
 vx31 - vy31 - vx56 + vy56 >= 1
 vx33 - vy33 - vx55 + vy55 >= 1
 vx33 - vy33 - vx56 + vy56 >= 1
 vx31 - vy31 - vx56 + vy56 >= 1
 vx31 - vy31 - vx57 + vy57 >= 1
 vx33 - vy33 - vx56 + vy56 >= 1
 vx33 - vy33 - vx57 + vy57 >= 1
 vx31 - vy31 - vx57 + vy57 >= 1
 vx31 - vy31 - vx58 + vy58 >= 1
 vx33 - vy33 - vx57 + vy57 >= 1
 vx33 - vy33 - vx58 + vy58 >= 1
 vx31 - vy31 - vx58 + vy58 >= 1
 vx31 - vy31 - vx59 + vy59 >= 1
 vx33 - vy33 - vx58 + vy58 >= 1
 vx33 - vy33 - vx59 + vy59 >= 1
 vx31 - vy31 - vx59 + vy59 >= 1
 vx31 - vy31 - vx60 + vy60 >= 1
 vx33 - vy33 - vx59 + vy59 >= 1
 vx33 - vy33 - vx60 + vy60 >= 1
 vx31 - vy31 - vx60 + vy60 >= 1
 vx31 - vy31 - vx61 + vy61 >= 1
 vx33 - vy33 - vx60 + vy60 >= 1
 vx33 - vy33 - vx61 + vy61 >= 1
 q25 - oa25 - ob25 - oc25 - od25 = 0
 a26 + a27 - 2 ua25 - oa25 = 0
 b26 + b27 - 2 ub25 - ob25 = 0
 c26 + c27 - 2 uc25 - oc25 = 0
 d26 + d27 - 2 ud25 - od25 = 0
 vx33 + vy33 - vx34 - vy34 <= -1
 vx33 + vy33 - vx32 - vy32 <= -1
 vx29 + vy29 - vx34 - vy34 <= -1
 vx29 + vy29 - vx32 - vy32 <= -1
 vx33 - vy33 - vx32 + vy32 >= 1
 vx33 - vy33 - vx65 + vy65 >= 1
 vx29 - vy29 - vx32 + vy32 >= 1
 vx29 - vy29 - vx65 + vy65 >= 1
 vx33 - vy33 - vx65 + vy65 >= 1
 vx33 - vy33 - vx28 + vy28 >= 1
 vx29 - vy29 - vx65 + vy65 >= 1
 vx29 - vy29 - vx28 + vy28 >= 1
 vx33 - vy33 - vx28 + vy28 >= 1
 vx33 - vy33 - vx18 + vy18 >= 1
 vx29 - vy29 - vx28 + vy28 >= 1
 vx29 - vy29 - vx18 + vy18 >= 1
 vx33 - vy33 - vx18 + vy18 >= 1
 vx33 - vy33 - vx19 + vy19 >= 1
 vx29 - vy29 - vx18 + vy18 >= 1
 vx29 - vy29 - vx19 + vy19 >= 1
 vx33 - vy33 - vx19 + vy19 >= 1
 vx33 - vy33 - vx26 + vy26 >= 1
 vx29 - vy29 - vx19 + vy19 >= 1
 vx29 - vy29 - vx26 + vy26 >= 1
 vx33 - vy33 - vx26 + vy26 >= 1
 vx33 - vy33 - vx21 + vy21 >= 1
 vx29 - vy29 - vx26 + vy26 >= 1
 vx29 - vy29 - vx21 + vy21 >= 1
 vx33 - vy33 - vx21 + vy21 >= 1
 vx33 - vy33 - vx24 + vy24 >= 1
 vx29 - vy29 - vx21 + vy21 >= 1
 vx29 - vy29 - vx24 + vy24 >= 1
 vx33 - vy33 - vx24 + vy24 >= 1
 vx33 - vy33 - vx23 + vy23 >= 1
 vx29 - vy29 - vx24 + vy24 >= 1
 vx29 - vy29 - vx23 + vy23 >= 1
 vx33 - vy33 - vx23 + vy23 >= 1
 vx33 - vy33 - vx20 + vy20 >= 1
 vx29 - vy29 - vx23 + vy23 >= 1
 vx29 - vy29 - vx20 + vy20 >= 1
 vx33 - vy33 - vx20 + vy20 >= 1
 vx33 - vy33 - vx25 + vy25 >= 1
 vx29 - vy29 - vx20 + vy20 >= 1
 vx29 - vy29 - vx25 + vy25 >= 1
 vx33 + vy33 - vx25 - vy25 <= -1
 vx33 + vy33 - vx22 - vy22 <= -1
 vx29 + vy29 - vx25 - vy25 <= -1
 vx29 + vy29 - vx22 - vy22 <= -1
 vx33 + vy33 - vx41 - vy41 <= -1
 vx33 + vy33 - vx64 - vy64 <= -1
 vx29 + vy29 - vx41 - vy41 <= -1
 vx29 + vy29 - vx64 - vy64 <= -1
 vx33 + vy33 - vx64 - vy64 <= -1
 vx33 + vy33 - vx43 - vy43 <= -1
 vx29 + vy29 - vx64 - vy64 <= -1
 vx29 + vy29 - vx43 - vy43 <= -1
 vx33 + vy33 - vx43 - vy43 <= -1
 vx33 + vy33 - vx44 - vy44 <= -1
 vx29 + vy29 - vx43 - vy43 <= -1
 vx29 + vy29 - vx44 - vy44 <= -1
 vx33 + vy33 - vx44 - vy44 <= -1
 vx33 + vy33 - vx3 - vy3 <= -1
 vx29 + vy29 - vx44 - vy44 <= -1
 vx29 + vy29 - vx3 - vy3 <= -1
 vx33 + vy33 - vx3 - vy3 <= -1
 vx33 + vy33 - vx45 - vy45 <= -1
 vx29 + vy29 - vx3 - vy3 <= -1
 vx29 + vy29 - vx45 - vy45 <= -1
 vx33 + vy33 - vx45 - vy45 <= -1
 vx33 + vy33 - vx46 - vy46 <= -1
 vx29 + vy29 - vx45 - vy45 <= -1
 vx29 + vy29 - vx46 - vy46 <= -1
 vx33 + vy33 - vx46 - vy46 <= -1
 vx33 + vy33 - vx47 - vy47 <= -1
 vx29 + vy29 - vx46 - vy46 <= -1
 vx29 + vy29 - vx47 - vy47 <= -1
 vx33 + vy33 - vx47 - vy47 <= -1
 vx33 + vy33 - vx48 - vy48 <= -1
 vx29 + vy29 - vx47 - vy47 <= -1
 vx29 + vy29 - vx48 - vy48 <= -1
 vy33 - vy48 <= -1
 vy33 - vy35 <= -1
 vy29 - vy48 <= -1
 vy29 - vy35 <= -1
 vx33 - vy33 - vx35 + vy35 >= 1
 vx33 - vy33 - vx49 + vy49 >= 1
 vx29 - vy29 - vx35 + vy35 >= 1
 vx29 - vy29 - vx49 + vy49 >= 1
 vx33 - vy33 - vx49 + vy49 >= 1
 vx33 - vy33 - vx50 + vy50 >= 1
 vx29 - vy29 - vx49 + vy49 >= 1
 vx29 - vy29 - vx50 + vy50 >= 1
 vx33 - vy33 - vx50 + vy50 >= 1
 vx33 - vy33 - vx51 + vy51 >= 1
 vx29 - vy29 - vx50 + vy50 >= 1
 vx29 - vy29 - vx51 + vy51 >= 1
 vx33 - vy33 - vx51 + vy51 >= 1
 vx33 - vy33 - vx52 + vy52 >= 1
 vx29 - vy29 - vx51 + vy51 >= 1
 vx29 - vy29 - vx52 + vy52 >= 1
 vx33 - vy33 - vx52 + vy52 >= 1
 vx33 - vy33 - vx62 + vy62 >= 1
 vx29 - vy29 - vx52 + vy52 >= 1
 vx29 - vy29 - vx62 + vy62 >= 1
 vx33 - vy33 - vx62 + vy62 >= 1
 vx33 - vy33 - vx53 + vy53 >= 1
 vx29 - vy29 - vx62 + vy62 >= 1
 vx29 - vy29 - vx53 + vy53 >= 1
 vx33 - vy33 - vx53 + vy53 >= 1
 vx33 - vy33 - vx54 + vy54 >= 1
 vx29 - vy29 - vx53 + vy53 >= 1
 vx29 - vy29 - vx54 + vy54 >= 1
 vx33 - vy33 - vx54 + vy54 >= 1
 vx33 - vy33 - vx55 + vy55 >= 1
 vx29 - vy29 - vx54 + vy54 >= 1
 vx29 - vy29 - vx55 + vy55 >= 1
 vx33 - vy33 - vx55 + vy55 >= 1
 vx33 - vy33 - vx56 + vy56 >= 1
 vx29 - vy29 - vx55 + vy55 >= 1
 vx29 - vy29 - vx56 + vy56 >= 1
 vx33 - vy33 - vx56 + vy56 >= 1
 vx33 - vy33 - vx57 + vy57 >= 1
 vx29 - vy29 - vx56 + vy56 >= 1
 vx29 - vy29 - vx57 + vy57 >= 1
 vx33 - vy33 - vx57 + vy57 >= 1
 vx33 - vy33 - vx58 + vy58 >= 1
 vx29 - vy29 - vx57 + vy57 >= 1
 vx29 - vy29 - vx58 + vy58 >= 1
 vx33 - vy33 - vx58 + vy58 >= 1
 vx33 - vy33 - vx59 + vy59 >= 1
 vx29 - vy29 - vx58 + vy58 >= 1
 vx29 - vy29 - vx59 + vy59 >= 1
 vx33 - vy33 - vx59 + vy59 >= 1
 vx33 - vy33 - vx60 + vy60 >= 1
 vx29 - vy29 - vx59 + vy59 >= 1
 vx29 - vy29 - vx60 + vy60 >= 1
 vx33 - vy33 - vx60 + vy60 >= 1
 vx33 - vy33 - vx61 + vy61 >= 1
 vx29 - vy29 - vx60 + vy60 >= 1
 vx29 - vy29 - vx61 + vy61 >= 1
 q26 - oa26 - ob26 - oc26 - od26 = 0
 a27 + a28 - 2 ua26 - oa26 = 0
 b27 + b28 - 2 ub26 - ob26 = 0
 c27 + c28 - 2 uc26 - oc26 = 0
 d27 + d28 - 2 ud26 - od26 = 0
 vx29 - vy29 - vx32 + vy32 >= 1
 vx29 - vy29 - vx65 + vy65 >= 1
 vx34 - vy34 - vx32 + vy32 >= 1
 vx34 - vy34 - vx65 + vy65 >= 1
 vx29 - vy29 - vx65 + vy65 >= 1
 vx29 - vy29 - vx28 + vy28 >= 1
 vx34 - vy34 - vx65 + vy65 >= 1
 vx34 - vy34 - vx28 + vy28 >= 1
 vx29 - vy29 - vx28 + vy28 >= 1
 vx29 - vy29 - vx18 + vy18 >= 1
 vx34 - vy34 - vx28 + vy28 >= 1
 vx34 - vy34 - vx18 + vy18 >= 1
 vx29 - vy29 - vx18 + vy18 >= 1
 vx29 - vy29 - vx19 + vy19 >= 1
 vx34 - vy34 - vx18 + vy18 >= 1
 vx34 - vy34 - vx19 + vy19 >= 1
 vx29 - vy29 - vx19 + vy19 >= 1
 vx29 - vy29 - vx26 + vy26 >= 1
 vx34 - vy34 - vx19 + vy19 >= 1
 vx34 - vy34 - vx26 + vy26 >= 1
 vx29 - vy29 - vx26 + vy26 >= 1
 vx29 - vy29 - vx21 + vy21 >= 1
 vx34 - vy34 - vx26 + vy26 >= 1
 vx34 - vy34 - vx21 + vy21 >= 1
 vx29 - vy29 - vx21 + vy21 >= 1
 vx29 - vy29 - vx24 + vy24 >= 1
 vx34 - vy34 - vx21 + vy21 >= 1
 vx34 - vy34 - vx24 + vy24 >= 1
 vx29 - vy29 - vx24 + vy24 >= 1
 vx29 - vy29 - vx23 + vy23 >= 1
 vx34 - vy34 - vx24 + vy24 >= 1
 vx34 - vy34 - vx23 + vy23 >= 1
 vx29 - vy29 - vx23 + vy23 >= 1
 vx29 - vy29 - vx20 + vy20 >= 1
 vx34 - vy34 - vx23 + vy23 >= 1
 vx34 - vy34 - vx20 + vy20 >= 1
 vx29 - vy29 - vx20 + vy20 >= 1
 vx29 - vy29 - vx25 + vy25 >= 1
 vx34 - vy34 - vx20 + vy20 >= 1
 vx34 - vy34 - vx25 + vy25 >= 1
 vx29 + vy29 - vx25 - vy25 <= -1
 vx29 + vy29 - vx22 - vy22 <= -1
 vx34 + vy34 - vx25 - vy25 <= -1
 vx34 + vy34 - vx22 - vy22 <= -1
 vx29 + vy29 - vx41 - vy41 <= -1
 vx29 + vy29 - vx64 - vy64 <= -1
 vx34 + vy34 - vx41 - vy41 <= -1
 vx34 + vy34 - vx64 - vy64 <= -1
 vx29 + vy29 - vx64 - vy64 <= -1
 vx29 + vy29 - vx43 - vy43 <= -1
 vx34 + vy34 - vx64 - vy64 <= -1
 vx34 + vy34 - vx43 - vy43 <= -1
 vx29 + vy29 - vx43 - vy43 <= -1
 vx29 + vy29 - vx44 - vy44 <= -1
 vx34 + vy34 - vx43 - vy43 <= -1
 vx34 + vy34 - vx44 - vy44 <= -1
 vx29 + vy29 - vx44 - vy44 <= -1
 vx29 + vy29 - vx3 - vy3 <= -1
 vx34 + vy34 - vx44 - vy44 <= -1
 vx34 + vy34 - vx3 - vy3 <= -1
 vx29 + vy29 - vx3 - vy3 <= -1
 vx29 + vy29 - vx45 - vy45 <= -1
 vx34 + vy34 - vx3 - vy3 <= -1
 vx34 + vy34 - vx45 - vy45 <= -1
 vx29 + vy29 - vx45 - vy45 <= -1
 vx29 + vy29 - vx46 - vy46 <= -1
 vx34 + vy34 - vx45 - vy45 <= -1
 vx34 + vy34 - vx46 - vy46 <= -1
 vx29 + vy29 - vx46 - vy46 <= -1
 vx29 + vy29 - vx47 - vy47 <= -1
 vx34 + vy34 - vx46 - vy46 <= -1
 vx34 + vy34 - vx47 - vy47 <= -1
 vx29 + vy29 - vx47 - vy47 <= -1
 vx29 + vy29 - vx48 - vy48 <= -1
 vx34 + vy34 - vx47 - vy47 <= -1
 vx34 + vy34 - vx48 - vy48 <= -1
 vy29 - vy48 <= -1
 vy29 - vy35 <= -1
 vy34 - vy48 <= -1
 vy34 - vy35 <= -1
 vx29 - vy29 - vx35 + vy35 >= 1
 vx29 - vy29 - vx49 + vy49 >= 1
 vx34 - vy34 - vx35 + vy35 >= 1
 vx34 - vy34 - vx49 + vy49 >= 1
 vx29 - vy29 - vx49 + vy49 >= 1
 vx29 - vy29 - vx50 + vy50 >= 1
 vx34 - vy34 - vx49 + vy49 >= 1
 vx34 - vy34 - vx50 + vy50 >= 1
 vx29 - vy29 - vx50 + vy50 >= 1
 vx29 - vy29 - vx51 + vy51 >= 1
 vx34 - vy34 - vx50 + vy50 >= 1
 vx34 - vy34 - vx51 + vy51 >= 1
 vx29 - vy29 - vx51 + vy51 >= 1
 vx29 - vy29 - vx52 + vy52 >= 1
 vx34 - vy34 - vx51 + vy51 >= 1
 vx34 - vy34 - vx52 + vy52 >= 1
 vx29 - vy29 - vx52 + vy52 >= 1
 vx29 - vy29 - vx62 + vy62 >= 1
 vx34 - vy34 - vx52 + vy52 >= 1
 vx34 - vy34 - vx62 + vy62 >= 1
 vx29 - vy29 - vx62 + vy62 >= 1
 vx29 - vy29 - vx53 + vy53 >= 1
 vx34 - vy34 - vx62 + vy62 >= 1
 vx34 - vy34 - vx53 + vy53 >= 1
 vx29 - vy29 - vx53 + vy53 >= 1
 vx29 - vy29 - vx54 + vy54 >= 1
 vx34 - vy34 - vx53 + vy53 >= 1
 vx34 - vy34 - vx54 + vy54 >= 1
 vx29 - vy29 - vx54 + vy54 >= 1
 vx29 - vy29 - vx55 + vy55 >= 1
 vx34 - vy34 - vx54 + vy54 >= 1
 vx34 - vy34 - vx55 + vy55 >= 1
 vx29 - vy29 - vx55 + vy55 >= 1
 vx29 - vy29 - vx56 + vy56 >= 1
 vx34 - vy34 - vx55 + vy55 >= 1
 vx34 - vy34 - vx56 + vy56 >= 1
 vx29 - vy29 - vx56 + vy56 >= 1
 vx29 - vy29 - vx57 + vy57 >= 1
 vx34 - vy34 - vx56 + vy56 >= 1
 vx34 - vy34 - vx57 + vy57 >= 1
 vx29 - vy29 - vx57 + vy57 >= 1
 vx29 - vy29 - vx58 + vy58 >= 1
 vx34 - vy34 - vx57 + vy57 >= 1
 vx34 - vy34 - vx58 + vy58 >= 1
 vx29 - vy29 - vx58 + vy58 >= 1
 vx29 - vy29 - vx59 + vy59 >= 1
 vx34 - vy34 - vx58 + vy58 >= 1
 vx34 - vy34 - vx59 + vy59 >= 1
 vx29 - vy29 - vx59 + vy59 >= 1
 vx29 - vy29 - vx60 + vy60 >= 1
 vx34 - vy34 - vx59 + vy59 >= 1
 vx34 - vy34 - vx60 + vy60 >= 1
 vx29 - vy29 - vx60 + vy60 >= 1
 vx29 - vy29 - vx61 + vy61 >= 1
 vx34 - vy34 - vx60 + vy60 >= 1
 vx34 - vy34 - vx61 + vy61 >= 1
 q27 - oa27 - ob27 - oc27 - od27 = 0
 a28 + a29 - 2 ua27 - oa27 = 0
 b28 + b29 - 2 ub27 - ob27 = 0
 c28 + c29 - 2 uc27 - oc27 = 0
 d28 + d29 - 2 ud27 - od27 = 0
 vx34 + vy34 - vx65 - vy65 <= -1
 vx34 + vy34 - vx28 - vy28 <= -1
 vx32 + vy32 - vx65 - vy65 <= -1
 vx32 + vy32 - vx28 - vy28 <= -1
 vx34 + vy34 - vx28 - vy28 <= -1
 vx34 + vy34 - vx18 - vy18 <= -1
 vx32 + vy32 - vx28 - vy28 <= -1
 vx32 + vy32 - vx18 - vy18 <= -1
 vx34 - vy34 - vx18 + vy18 >= 1
 vx34 - vy34 - vx19 + vy19 >= 1
 vx32 - vy32 - vx18 + vy18 >= 1
 vx32 - vy32 - vx19 + vy19 >= 1
 vx34 - vy34 - vx19 + vy19 >= 1
 vx34 - vy34 - vx26 + vy26 >= 1
 vx32 - vy32 - vx19 + vy19 >= 1
 vx32 - vy32 - vx26 + vy26 >= 1
 vx34 - vy34 - vx26 + vy26 >= 1
 vx34 - vy34 - vx21 + vy21 >= 1
 vx32 - vy32 - vx26 + vy26 >= 1
 vx32 - vy32 - vx21 + vy21 >= 1
 vx34 - vy34 - vx21 + vy21 >= 1
 vx34 - vy34 - vx24 + vy24 >= 1
 vx32 - vy32 - vx21 + vy21 >= 1
 vx32 - vy32 - vx24 + vy24 >= 1
 vx34 - vy34 - vx24 + vy24 >= 1
 vx34 - vy34 - vx23 + vy23 >= 1
 vx32 - vy32 - vx24 + vy24 >= 1
 vx32 - vy32 - vx23 + vy23 >= 1
 vx34 - vy34 - vx23 + vy23 >= 1
 vx34 - vy34 - vx20 + vy20 >= 1
 vx32 - vy32 - vx23 + vy23 >= 1
 vx32 - vy32 - vx20 + vy20 >= 1
 vx34 - vy34 - vx20 + vy20 >= 1
 vx34 - vy34 - vx25 + vy25 >= 1
 vx32 - vy32 - vx20 + vy20 >= 1
 vx32 - vy32 - vx25 + vy25 >= 1
 vx34 + vy34 - vx25 - vy25 <= -1
 vx34 + vy34 - vx22 - vy22 <= -1
 vx32 + vy32 - vx25 - vy25 <= -1
 vx32 + vy32 - vx22 - vy22 <= -1
 vx34 - vx41 <= -1
 vx34 - vx64 <= -1
 vx32 - vx41 <= -1
 vx32 - vx64 <= -1
 vx34 + vy34 - vx64 - vy64 <= -1
 vx34 + vy34 - vx43 - vy43 <= -1
 vx32 + vy32 - vx64 - vy64 <= -1
 vx32 + vy32 - vx43 - vy43 <= -1
 vx34 + vy34 - vx43 - vy43 <= -1
 vx34 + vy34 - vx44 - vy44 <= -1
 vx32 + vy32 - vx43 - vy43 <= -1
 vx32 + vy32 - vx44 - vy44 <= -1
 vx34 + vy34 - vx44 - vy44 <= -1
 vx34 + vy34 - vx3 - vy3 <= -1
 vx32 + vy32 - vx44 - vy44 <= -1
 vx32 + vy32 - vx3 - vy3 <= -1
 vx34 + vy34 - vx3 - vy3 <= -1
 vx34 + vy34 - vx45 - vy45 <= -1
 vx32 + vy32 - vx3 - vy3 <= -1
 vx32 + vy32 - vx45 - vy45 <= -1
 vx34 + vy34 - vx45 - vy45 <= -1
 vx34 + vy34 - vx46 - vy46 <= -1
 vx32 + vy32 - vx45 - vy45 <= -1
 vx32 + vy32 - vx46 - vy46 <= -1
 vx34 + vy34 - vx46 - vy46 <= -1
 vx34 + vy34 - vx47 - vy47 <= -1
 vx32 + vy32 - vx46 - vy46 <= -1
 vx32 + vy32 - vx47 - vy47 <= -1
 vx34 + vy34 - vx47 - vy47 <= -1
 vx34 + vy34 - vx48 - vy48 <= -1
 vx32 + vy32 - vx47 - vy47 <= -1
 vx32 + vy32 - vx48 - vy48 <= -1
 vx34 + vy34 - vx48 - vy48 <= -1
 vx34 + vy34 - vx35 - vy35 <= -1
 vx32 + vy32 - vx48 - vy48 <= -1
 vx32 + vy32 - vx35 - vy35 <= -1
 vy34 - vy35 <= -1
 vy34 - vy49 <= -1
 vy32 - vy35 <= -1
 vy32 - vy49 <= -1
 vx34 - vy34 - vx49 + vy49 >= 1
 vx34 - vy34 - vx50 + vy50 >= 1
 vx32 - vy32 - vx49 + vy49 >= 1
 vx32 - vy32 - vx50 + vy50 >= 1
 vx34 - vy34 - vx50 + vy50 >= 1
 vx34 - vy34 - vx51 + vy51 >= 1
 vx32 - vy32 - vx50 + vy50 >= 1
 vx32 - vy32 - vx51 + vy51 >= 1
 vx34 - vy34 - vx51 + vy51 >= 1
 vx34 - vy34 - vx52 + vy52 >= 1
 vx32 - vy32 - vx51 + vy51 >= 1
 vx32 - vy32 - vx52 + vy52 >= 1
 vx34 - vy34 - vx52 + vy52 >= 1
 vx34 - vy34 - vx62 + vy62 >= 1
 vx32 - vy32 - vx52 + vy52 >= 1
 vx32 - vy32 - vx62 + vy62 >= 1
 vx34 - vy34 - vx62 + vy62 >= 1
 vx34 - vy34 - vx53 + vy53 >= 1
 vx32 - vy32 - vx62 + vy62 >= 1
 vx32 - vy32 - vx53 + vy53 >= 1
 vx34 - vy34 - vx53 + vy53 >= 1
 vx34 - vy34 - vx54 + vy54 >= 1
 vx32 - vy32 - vx53 + vy53 >= 1
 vx32 - vy32 - vx54 + vy54 >= 1
 vx34 - vy34 - vx54 + vy54 >= 1
 vx34 - vy34 - vx55 + vy55 >= 1
 vx32 - vy32 - vx54 + vy54 >= 1
 vx32 - vy32 - vx55 + vy55 >= 1
 vx34 - vy34 - vx55 + vy55 >= 1
 vx34 - vy34 - vx56 + vy56 >= 1
 vx32 - vy32 - vx55 + vy55 >= 1
 vx32 - vy32 - vx56 + vy56 >= 1
 vx34 - vy34 - vx56 + vy56 >= 1
 vx34 - vy34 - vx57 + vy57 >= 1
 vx32 - vy32 - vx56 + vy56 >= 1
 vx32 - vy32 - vx57 + vy57 >= 1
 vx34 - vy34 - vx57 + vy57 >= 1
 vx34 - vy34 - vx58 + vy58 >= 1
 vx32 - vy32 - vx57 + vy57 >= 1
 vx32 - vy32 - vx58 + vy58 >= 1
 vx34 - vy34 - vx58 + vy58 >= 1
 vx34 - vy34 - vx59 + vy59 >= 1
 vx32 - vy32 - vx58 + vy58 >= 1
 vx32 - vy32 - vx59 + vy59 >= 1
 vx34 - vy34 - vx59 + vy59 >= 1
 vx34 - vy34 - vx60 + vy60 >= 1
 vx32 - vy32 - vx59 + vy59 >= 1
 vx32 - vy32 - vx60 + vy60 >= 1
 vx34 - vy34 - vx60 + vy60 >= 1
 vx34 - vy34 - vx61 + vy61 >= 1
 vx32 - vy32 - vx60 + vy60 >= 1
 vx32 - vy32 - vx61 + vy61 >= 1
 q28 - oa28 - ob28 - oc28 - od28 = 0
 a29 + a30 - 2 ua28 - oa28 = 0
 b29 + b30 - 2 ub28 - ob28 = 0
 c29 + c30 - 2 uc28 - oc28 = 0
 d29 + d30 - 2 ud28 - od28 = 0
 vx32 - vy32 - vx28 + vy28 >= 1
 vx32 - vy32 - vx18 + vy18 >= 1
 vx65 - vy65 - vx28 + vy28 >= 1
 vx65 - vy65 - vx18 + vy18 >= 1
 vx32 - vy32 - vx18 + vy18 >= 1
 vx32 - vy32 - vx19 + vy19 >= 1
 vx65 - vy65 - vx18 + vy18 >= 1
 vx65 - vy65 - vx19 + vy19 >= 1
 vx32 - vy32 - vx19 + vy19 >= 1
 vx32 - vy32 - vx26 + vy26 >= 1
 vx65 - vy65 - vx19 + vy19 >= 1
 vx65 - vy65 - vx26 + vy26 >= 1
 vx32 - vy32 - vx26 + vy26 >= 1
 vx32 - vy32 - vx21 + vy21 >= 1
 vx65 - vy65 - vx26 + vy26 >= 1
 vx65 - vy65 - vx21 + vy21 >= 1
 vx32 - vy32 - vx21 + vy21 >= 1
 vx32 - vy32 - vx24 + vy24 >= 1
 vx65 - vy65 - vx21 + vy21 >= 1
 vx65 - vy65 - vx24 + vy24 >= 1
 vx32 - vy32 - vx24 + vy24 >= 1
 vx32 - vy32 - vx23 + vy23 >= 1
 vx65 - vy65 - vx24 + vy24 >= 1
 vx65 - vy65 - vx23 + vy23 >= 1
 vx32 - vy32 - vx23 + vy23 >= 1
 vx32 - vy32 - vx20 + vy20 >= 1
 vx65 - vy65 - vx23 + vy23 >= 1
 vx65 - vy65 - vx20 + vy20 >= 1
 vx32 - vy32 - vx20 + vy20 >= 1
 vx32 - vy32 - vx25 + vy25 >= 1
 vx65 - vy65 - vx20 + vy20 >= 1
 vx65 - vy65 - vx25 + vy25 >= 1
 vx32 + vy32 - vx25 - vy25 <= -1
 vx32 + vy32 - vx22 - vy22 <= -1
 vx65 + vy65 - vx25 - vy25 <= -1
 vx65 + vy65 - vx22 - vy22 <= -1
 vx32 - vy32 - vx41 + vy41 <= -1
 vx32 - vy32 - vx64 + vy64 <= -1
 vx65 - vy65 - vx41 + vy41 <= -1
 vx65 - vy65 - vx64 + vy64 <= -1
 vx32 - vy32 - vx64 + vy64 <= -1
 vx32 - vy32 - vx43 + vy43 <= -1
 vx65 - vy65 - vx64 + vy64 <= -1
 vx65 - vy65 - vx43 + vy43 <= -1
 vx32 - vx43 <= -1
 vx32 - vx44 <= -1
 vx65 - vx43 <= -1
 vx65 - vx44 <= -1
 vx32 - vx44 <= -1
 vx32 - vx3 <= -1
 vx65 - vx44 <= -1
 vx65 - vx3 <= -1
 vx32 - vx3 <= -1
 vx32 - vx45 <= -1
 vx65 - vx3 <= -1
 vx65 - vx45 <= -1
 vx32 - vx45 <= -1
 vx32 - vx46 <= -1
 vx65 - vx45 <= -1
 vx65 - vx46 <= -1
 vx32 - vx46 <= -1
 vx32 - vx47 <= -1
 vx65 - vx46 <= -1
 vx65 - vx47 <= -1
 vx32 + vy32 - vx47 - vy47 <= -1
 vx32 + vy32 - vx48 - vy48 <= -1
 vx65 + vy65 - vx47 - vy47 <= -1
 vx65 + vy65 - vx48 - vy48 <= -1
 vx32 - vx48 <= -1
 vx32 - vx35 <= -1
 vx65 - vx48 <= -1
 vx65 - vx35 <= -1
 vx32 - vy32 - vx49 + vy49 >= 1
 vx32 - vy32 - vx50 + vy50 >= 1
 vx65 - vy65 - vx49 + vy49 >= 1
 vx65 - vy65 - vx50 + vy50 >= 1
 vx32 - vy32 - vx50 + vy50 >= 1
 vx32 - vy32 - vx51 + vy51 >= 1
 vx65 - vy65 - vx50 + vy50 >= 1
 vx65 - vy65 - vx51 + vy51 >= 1
 vx32 - vy32 - vx51 + vy51 >= 1
 vx32 - vy32 - vx52 + vy52 >= 1
 vx65 - vy65 - vx51 + vy51 >= 1
 vx65 - vy65 - vx52 + vy52 >= 1
 vx32 - vy32 - vx52 + vy52 >= 1
 vx32 - vy32 - vx62 + vy62 >= 1
 vx65 - vy65 - vx52 + vy52 >= 1
 vx65 - vy65 - vx62 + vy62 >= 1
 vx32 - vy32 - vx62 + vy62 >= 1
 vx32 - vy32 - vx53 + vy53 >= 1
 vx65 - vy65 - vx62 + vy62 >= 1
 vx65 - vy65 - vx53 + vy53 >= 1
 vx32 - vy32 - vx53 + vy53 >= 1
 vx32 - vy32 - vx54 + vy54 >= 1
 vx65 - vy65 - vx53 + vy53 >= 1
 vx65 - vy65 - vx54 + vy54 >= 1
 vx32 - vy32 - vx54 + vy54 >= 1
 vx32 - vy32 - vx55 + vy55 >= 1
 vx65 - vy65 - vx54 + vy54 >= 1
 vx65 - vy65 - vx55 + vy55 >= 1
 vx32 - vy32 - vx55 + vy55 >= 1
 vx32 - vy32 - vx56 + vy56 >= 1
 vx65 - vy65 - vx55 + vy55 >= 1
 vx65 - vy65 - vx56 + vy56 >= 1
 vx32 - vy32 - vx56 + vy56 >= 1
 vx32 - vy32 - vx57 + vy57 >= 1
 vx65 - vy65 - vx56 + vy56 >= 1
 vx65 - vy65 - vx57 + vy57 >= 1
 vx32 - vy32 - vx57 + vy57 >= 1
 vx32 - vy32 - vx58 + vy58 >= 1
 vx65 - vy65 - vx57 + vy57 >= 1
 vx65 - vy65 - vx58 + vy58 >= 1
 vx32 - vy32 - vx58 + vy58 >= 1
 vx32 - vy32 - vx59 + vy59 >= 1
 vx65 - vy65 - vx58 + vy58 >= 1
 vx65 - vy65 - vx59 + vy59 >= 1
 vx32 - vy32 - vx59 + vy59 >= 1
 vx32 - vy32 - vx60 + vy60 >= 1
 vx65 - vy65 - vx59 + vy59 >= 1
 vx65 - vy65 - vx60 + vy60 >= 1
 vx32 - vy32 - vx60 + vy60 >= 1
 vx32 - vy32 - vx61 + vy61 >= 1
 vx65 - vy65 - vx60 + vy60 >= 1
 vx65 - vy65 - vx61 + vy61 >= 1
 q29 - oa29 - ob29 - oc29 - od29 = 0
 a30 + a31 - 2 ua29 - oa29 = 0
 b30 + b31 - 2 ub29 - ob29 = 0
 c30 + c31 - 2 uc29 - oc29 = 0
 d30 + d31 - 2 ud29 - od29 = 0
 vx65 - vy65 - vx18 + vy18 >= 1
 vx65 - vy65 - vx19 + vy19 >= 1
 vx28 - vy28 - vx18 + vy18 >= 1
 vx28 - vy28 - vx19 + vy19 >= 1
 vx65 - vy65 - vx19 + vy19 >= 1
 vx65 - vy65 - vx26 + vy26 >= 1
 vx28 - vy28 - vx19 + vy19 >= 1
 vx28 - vy28 - vx26 + vy26 >= 1
 vx65 - vy65 - vx26 + vy26 >= 1
 vx65 - vy65 - vx21 + vy21 >= 1
 vx28 - vy28 - vx26 + vy26 >= 1
 vx28 - vy28 - vx21 + vy21 >= 1
 vx65 - vy65 - vx21 + vy21 >= 1
 vx65 - vy65 - vx24 + vy24 >= 1
 vx28 - vy28 - vx21 + vy21 >= 1
 vx28 - vy28 - vx24 + vy24 >= 1
 vx65 - vy65 - vx24 + vy24 >= 1
 vx65 - vy65 - vx23 + vy23 >= 1
 vx28 - vy28 - vx24 + vy24 >= 1
 vx28 - vy28 - vx23 + vy23 >= 1
 vx65 - vy65 - vx23 + vy23 >= 1
 vx65 - vy65 - vx20 + vy20 >= 1
 vx28 - vy28 - vx23 + vy23 >= 1
 vx28 - vy28 - vx20 + vy20 >= 1
 vx65 - vy65 - vx20 + vy20 >= 1
 vx65 - vy65 - vx25 + vy25 >= 1
 vx28 - vy28 - vx20 + vy20 >= 1
 vx28 - vy28 - vx25 + vy25 >= 1
 vx65 + vy65 - vx25 - vy25 <= -1
 vx65 + vy65 - vx22 - vy22 <= -1
 vx28 + vy28 - vx25 - vy25 <= -1
 vx28 + vy28 - vx22 - vy22 <= -1
 vx65 - vy65 - vx41 + vy41 <= -1
 vx65 - vy65 - vx64 + vy64 <= -1
 vx28 - vy28 - vx41 + vy41 <= -1
 vx28 - vy28 - vx64 + vy64 <= -1
 vx65 - vy65 - vx64 + vy64 <= -1
 vx65 - vy65 - vx43 + vy43 <= -1
 vx28 - vy28 - vx64 + vy64 <= -1
 vx28 - vy28 - vx43 + vy43 <= -1
 vx65 - vy65 - vx43 + vy43 <= -1
 vx65 - vy65 - vx44 + vy44 <= -1
 vx28 - vy28 - vx43 + vy43 <= -1
 vx28 - vy28 - vx44 + vy44 <= -1
 vx65 - vy65 - vx44 + vy44 <= -1
 vx65 - vy65 - vx3 + vy3 <= -1
 vx28 - vy28 - vx44 + vy44 <= -1
 vx28 - vy28 - vx3 + vy3 <= -1
 vx65 - vy65 - vx3 + vy3 <= -1
 vx65 - vy65 - vx45 + vy45 <= -1
 vx28 - vy28 - vx3 + vy3 <= -1
 vx28 - vy28 - vx45 + vy45 <= -1
 vx65 - vy65 - vx45 + vy45 <= -1
 vx65 - vy65 - vx46 + vy46 <= -1
 vx28 - vy28 - vx45 + vy45 <= -1
 vx28 - vy28 - vx46 + vy46 <= -1
 vx65 - vy65 - vx46 + vy46 <= -1
 vx65 - vy65 - vx47 + vy47 <= -1
 vx28 - vy28 - vx46 + vy46 <= -1
 vx28 - vy28 - vx47 + vy47 <= -1
 vx65 - vx47 <= -1
 vx65 - vx48 <= -1
 vx28 - vx47 <= -1
 vx28 - vx48 <= -1
 vx65 - vy65 - vx48 + vy48 <= -1
 vx65 - vy65 - vx35 + vy35 <= -1
 vx28 - vy28 - vx48 + vy48 <= -1
 vx28 - vy28 - vx35 + vy35 <= -1
 vx65 + vy65 - vx35 - vy35 >= 1
 vx65 + vy65 - vx49 - vy49 >= 1
 vx28 + vy28 - vx35 - vy35 >= 1
 vx28 + vy28 - vx49 - vy49 >= 1
 vx65 - vx49 >= 1
 vx65 - vx50 >= 1
 vx28 - vx49 >= 1
 vx28 - vx50 >= 1
 vx65 - vy65 - vx50 + vy50 >= 1
 vx65 - vy65 - vx51 + vy51 >= 1
 vx28 - vy28 - vx50 + vy50 >= 1
 vx28 - vy28 - vx51 + vy51 >= 1
 vx65 - vy65 - vx51 + vy51 >= 1
 vx65 - vy65 - vx52 + vy52 >= 1
 vx28 - vy28 - vx51 + vy51 >= 1
 vx28 - vy28 - vx52 + vy52 >= 1
 vx65 - vy65 - vx52 + vy52 >= 1
 vx65 - vy65 - vx62 + vy62 >= 1
 vx28 - vy28 - vx52 + vy52 >= 1
 vx28 - vy28 - vx62 + vy62 >= 1
 vx65 - vy65 - vx62 + vy62 >= 1
 vx65 - vy65 - vx53 + vy53 >= 1
 vx28 - vy28 - vx62 + vy62 >= 1
 vx28 - vy28 - vx53 + vy53 >= 1
 vx65 - vy65 - vx53 + vy53 >= 1
 vx65 - vy65 - vx54 + vy54 >= 1
 vx28 - vy28 - vx53 + vy53 >= 1
 vx28 - vy28 - vx54 + vy54 >= 1
 vx65 - vy65 - vx54 + vy54 >= 1
 vx65 - vy65 - vx55 + vy55 >= 1
 vx28 - vy28 - vx54 + vy54 >= 1
 vx28 - vy28 - vx55 + vy55 >= 1
 vx65 - vy65 - vx55 + vy55 >= 1
 vx65 - vy65 - vx56 + vy56 >= 1
 vx28 - vy28 - vx55 + vy55 >= 1
 vx28 - vy28 - vx56 + vy56 >= 1
 vx65 - vy65 - vx56 + vy56 >= 1
 vx65 - vy65 - vx57 + vy57 >= 1
 vx28 - vy28 - vx56 + vy56 >= 1
 vx28 - vy28 - vx57 + vy57 >= 1
 vx65 - vy65 - vx57 + vy57 >= 1
 vx65 - vy65 - vx58 + vy58 >= 1
 vx28 - vy28 - vx57 + vy57 >= 1
 vx28 - vy28 - vx58 + vy58 >= 1
 vx65 - vy65 - vx58 + vy58 >= 1
 vx65 - vy65 - vx59 + vy59 >= 1
 vx28 - vy28 - vx58 + vy58 >= 1
 vx28 - vy28 - vx59 + vy59 >= 1
 vx65 - vy65 - vx59 + vy59 >= 1
 vx65 - vy65 - vx60 + vy60 >= 1
 vx28 - vy28 - vx59 + vy59 >= 1
 vx28 - vy28 - vx60 + vy60 >= 1
 vx65 - vy65 - vx60 + vy60 >= 1
 vx65 - vy65 - vx61 + vy61 >= 1
 vx28 - vy28 - vx60 + vy60 >= 1
 vx28 - vy28 - vx61 + vy61 >= 1
 q30 - oa30 - ob30 - oc30 - od30 = 0
 a31 + a32 - 2 ua30 - oa30 = 0
 b31 + b32 - 2 ub30 - ob30 = 0
 c31 + c32 - 2 uc30 - oc30 = 0
 d31 + d32 - 2 ud30 - od30 = 0
 vx28 - vy28 - vx19 + vy19 >= 1
 vx28 - vy28 - vx26 + vy26 >= 1
 vx18 - vy18 - vx19 + vy19 >= 1
 vx18 - vy18 - vx26 + vy26 >= 1
 vx28 - vy28 - vx26 + vy26 >= 1
 vx28 - vy28 - vx21 + vy21 >= 1
 vx18 - vy18 - vx26 + vy26 >= 1
 vx18 - vy18 - vx21 + vy21 >= 1
 vx28 - vy28 - vx21 + vy21 >= 1
 vx28 - vy28 - vx24 + vy24 >= 1
 vx18 - vy18 - vx21 + vy21 >= 1
 vx18 - vy18 - vx24 + vy24 >= 1
 vx28 - vy28 - vx24 + vy24 >= 1
 vx28 - vy28 - vx23 + vy23 >= 1
 vx18 - vy18 - vx24 + vy24 >= 1
 vx18 - vy18 - vx23 + vy23 >= 1
 vx28 - vy28 - vx23 + vy23 >= 1
 vx28 - vy28 - vx20 + vy20 >= 1
 vx18 - vy18 - vx23 + vy23 >= 1
 vx18 - vy18 - vx20 + vy20 >= 1
 vx28 + vy28 - vx20 - vy20 <= -1
 vx28 + vy28 - vx25 - vy25 <= -1
 vx18 + vy18 - vx20 - vy20 <= -1
 vx18 + vy18 - vx25 - vy25 <= -1
 vx28 + vy28 - vx25 - vy25 <= -1
 vx28 + vy28 - vx22 - vy22 <= -1
 vx18 + vy18 - vx25 - vy25 <= -1
 vx18 + vy18 - vx22 - vy22 <= -1
 vx28 - vy28 - vx41 + vy41 <= -1
 vx28 - vy28 - vx64 + vy64 <= -1
 vx18 - vy18 - vx41 + vy41 <= -1
 vx18 - vy18 - vx64 + vy64 <= -1
 vx28 - vy28 - vx64 + vy64 <= -1
 vx28 - vy28 - vx43 + vy43 <= -1
 vx18 - vy18 - vx64 + vy64 <= -1
 vx18 - vy18 - vx43 + vy43 <= -1
 vx28 - vy28 - vx43 + vy43 <= -1
 vx28 - vy28 - vx44 + vy44 <= -1
 vx18 - vy18 - vx43 + vy43 <= -1
 vx18 - vy18 - vx44 + vy44 <= -1
 vx28 - vy28 - vx44 + vy44 <= -1
 vx28 - vy28 - vx3 + vy3 <= -1
 vx18 - vy18 - vx44 + vy44 <= -1
 vx18 - vy18 - vx3 + vy3 <= -1
 vx28 - vy28 - vx3 + vy3 <= -1
 vx28 - vy28 - vx45 + vy45 <= -1
 vx18 - vy18 - vx3 + vy3 <= -1
 vx18 - vy18 - vx45 + vy45 <= -1
 vx28 - vy28 - vx45 + vy45 <= -1
 vx28 - vy28 - vx46 + vy46 <= -1
 vx18 - vy18 - vx45 + vy45 <= -1
 vx18 - vy18 - vx46 + vy46 <= -1
 vx28 - vy28 - vx46 + vy46 <= -1
 vx28 - vy28 - vx47 + vy47 <= -1
 vx18 - vy18 - vx46 + vy46 <= -1
 vx18 - vy18 - vx47 + vy47 <= -1
 vx28 - vy28 - vx47 + vy47 <= -1
 vx28 - vy28 - vx48 + vy48 <= -1
 vx18 - vy18 - vx47 + vy47 <= -1
 vx18 - vy18 - vx48 + vy48 <= -1
 vx28 - vy28 - vx48 + vy48 <= -1
 vx28 - vy28 - vx35 + vy35 <= -1
 vx18 - vy18 - vx48 + vy48 <= -1
 vx18 - vy18 - vx35 + vy35 <= -1
 vx28 + vy28 - vx35 - vy35 >= 1
 vx28 + vy28 - vx49 - vy49 >= 1
 vx18 + vy18 - vx35 - vy35 >= 1
 vx18 + vy18 - vx49 - vy49 >= 1
 vx28 + vy28 - vx49 - vy49 >= 1
 vx28 + vy28 - vx50 - vy50 >= 1
 vx18 + vy18 - vx49 - vy49 >= 1
 vx18 + vy18 - vx50 - vy50 >= 1
 vx28 - vx50 >= 1
 vx28 - vx51 >= 1
 vx18 - vx50 >= 1
 vx18 - vx51 >= 1
 vx28 - vy28 - vx51 + vy51 >= 1
 vx28 - vy28 - vx52 + vy52 >= 1
 vx18 - vy18 - vx51 + vy51 >= 1
 vx18 - vy18 - vx52 + vy52 >= 1
 vx28 - vy28 - vx52 + vy52 >= 1
 vx28 - vy28 - vx62 + vy62 >= 1
 vx18 - vy18 - vx52 + vy52 >= 1
 vx18 - vy18 - vx62 + vy62 >= 1
 vx28 - vy28 - vx62 + vy62 >= 1
 vx28 - vy28 - vx53 + vy53 >= 1
 vx18 - vy18 - vx62 + vy62 >= 1
 vx18 - vy18 - vx53 + vy53 >= 1
 vx28 - vy28 - vx53 + vy53 >= 1
 vx28 - vy28 - vx54 + vy54 >= 1
 vx18 - vy18 - vx53 + vy53 >= 1
 vx18 - vy18 - vx54 + vy54 >= 1
 vx28 - vy28 - vx54 + vy54 >= 1
 vx28 - vy28 - vx55 + vy55 >= 1
 vx18 - vy18 - vx54 + vy54 >= 1
 vx18 - vy18 - vx55 + vy55 >= 1
 vx28 - vy28 - vx55 + vy55 >= 1
 vx28 - vy28 - vx56 + vy56 >= 1
 vx18 - vy18 - vx55 + vy55 >= 1
 vx18 - vy18 - vx56 + vy56 >= 1
 vx28 - vy28 - vx56 + vy56 >= 1
 vx28 - vy28 - vx57 + vy57 >= 1
 vx18 - vy18 - vx56 + vy56 >= 1
 vx18 - vy18 - vx57 + vy57 >= 1
 vx28 - vy28 - vx57 + vy57 >= 1
 vx28 - vy28 - vx58 + vy58 >= 1
 vx18 - vy18 - vx57 + vy57 >= 1
 vx18 - vy18 - vx58 + vy58 >= 1
 vx28 - vy28 - vx58 + vy58 >= 1
 vx28 - vy28 - vx59 + vy59 >= 1
 vx18 - vy18 - vx58 + vy58 >= 1
 vx18 - vy18 - vx59 + vy59 >= 1
 vx28 - vy28 - vx59 + vy59 >= 1
 vx28 - vy28 - vx60 + vy60 >= 1
 vx18 - vy18 - vx59 + vy59 >= 1
 vx18 - vy18 - vx60 + vy60 >= 1
 vx28 - vy28 - vx60 + vy60 >= 1
 vx28 - vy28 - vx61 + vy61 >= 1
 vx18 - vy18 - vx60 + vy60 >= 1
 vx18 - vy18 - vx61 + vy61 >= 1
 q31 - oa31 - ob31 - oc31 - od31 = 0
 a32 + a33 - 2 ua31 - oa31 = 0
 b32 + b33 - 2 ub31 - ob31 = 0
 c32 + c33 - 2 uc31 - oc31 = 0
 d32 + d33 - 2 ud31 - od31 = 0
 vx18 + vy18 - vx26 - vy26 <= -1
 vx18 + vy18 - vx21 - vy21 <= -1
 vx19 + vy19 - vx26 - vy26 <= -1
 vx19 + vy19 - vx21 - vy21 <= -1
 vx18 + vy18 - vx21 - vy21 <= -1
 vx18 + vy18 - vx24 - vy24 <= -1
 vx19 + vy19 - vx21 - vy21 <= -1
 vx19 + vy19 - vx24 - vy24 <= -1
 vx18 + vy18 - vx24 - vy24 <= -1
 vx18 + vy18 - vx23 - vy23 <= -1
 vx19 + vy19 - vx24 - vy24 <= -1
 vx19 + vy19 - vx23 - vy23 <= -1
 vx18 + vy18 - vx23 - vy23 <= -1
 vx18 + vy18 - vx20 - vy20 <= -1
 vx19 + vy19 - vx23 - vy23 <= -1
 vx19 + vy19 - vx20 - vy20 <= -1
 vx18 + vy18 - vx20 - vy20 <= -1
 vx18 + vy18 - vx25 - vy25 <= -1
 vx19 + vy19 - vx20 - vy20 <= -1
 vx19 + vy19 - vx25 - vy25 <= -1
 vx18 + vy18 - vx25 - vy25 <= -1
 vx18 + vy18 - vx22 - vy22 <= -1
 vx19 + vy19 - vx25 - vy25 <= -1
 vx19 + vy19 - vx22 - vy22 <= -1
 vx18 - vy18 - vx41 + vy41 <= -1
 vx18 - vy18 - vx64 + vy64 <= -1
 vx19 - vy19 - vx41 + vy41 <= -1
 vx19 - vy19 - vx64 + vy64 <= -1
 vx18 - vy18 - vx64 + vy64 <= -1
 vx18 - vy18 - vx43 + vy43 <= -1
 vx19 - vy19 - vx64 + vy64 <= -1
 vx19 - vy19 - vx43 + vy43 <= -1
 vx18 - vy18 - vx43 + vy43 <= -1
 vx18 - vy18 - vx44 + vy44 <= -1
 vx19 - vy19 - vx43 + vy43 <= -1
 vx19 - vy19 - vx44 + vy44 <= -1
 vx18 - vy18 - vx44 + vy44 <= -1
 vx18 - vy18 - vx3 + vy3 <= -1
 vx19 - vy19 - vx44 + vy44 <= -1
 vx19 - vy19 - vx3 + vy3 <= -1
 vx18 - vy18 - vx3 + vy3 <= -1
 vx18 - vy18 - vx45 + vy45 <= -1
 vx19 - vy19 - vx3 + vy3 <= -1
 vx19 - vy19 - vx45 + vy45 <= -1
 vx18 - vy18 - vx45 + vy45 <= -1
 vx18 - vy18 - vx46 + vy46 <= -1
 vx19 - vy19 - vx45 + vy45 <= -1
 vx19 - vy19 - vx46 + vy46 <= -1
 vx18 - vy18 - vx46 + vy46 <= -1
 vx18 - vy18 - vx47 + vy47 <= -1
 vx19 - vy19 - vx46 + vy46 <= -1
 vx19 - vy19 - vx47 + vy47 <= -1
 vx18 - vy18 - vx47 + vy47 <= -1
 vx18 - vy18 - vx48 + vy48 <= -1
 vx19 - vy19 - vx47 + vy47 <= -1
 vx19 - vy19 - vx48 + vy48 <= -1
 vx18 - vy18 - vx48 + vy48 <= -1
 vx18 - vy18 - vx35 + vy35 <= -1
 vx19 - vy19 - vx48 + vy48 <= -1
 vx19 - vy19 - vx35 + vy35 <= -1
 vx18 + vy18 - vx35 - vy35 >= 1
 vx18 + vy18 - vx49 - vy49 >= 1
 vx19 + vy19 - vx35 - vy35 >= 1
 vx19 + vy19 - vx49 - vy49 >= 1
 vx18 + vy18 - vx49 - vy49 >= 1
 vx18 + vy18 - vx50 - vy50 >= 1
 vx19 + vy19 - vx49 - vy49 >= 1
 vx19 + vy19 - vx50 - vy50 >= 1
 vx18 + vy18 - vx50 - vy50 >= 1
 vx18 + vy18 - vx51 - vy51 >= 1
 vx19 + vy19 - vx50 - vy50 >= 1
 vx19 + vy19 - vx51 - vy51 >= 1
 vx18 - vx51 >= 1
 vx18 - vx52 >= 1
 vx19 - vx51 >= 1
 vx19 - vx52 >= 1
 vx18 - vy18 - vx52 + vy52 >= 1
 vx18 - vy18 - vx62 + vy62 >= 1
 vx19 - vy19 - vx52 + vy52 >= 1
 vx19 - vy19 - vx62 + vy62 >= 1
 vx18 - vy18 - vx62 + vy62 >= 1
 vx18 - vy18 - vx53 + vy53 >= 1
 vx19 - vy19 - vx62 + vy62 >= 1
 vx19 - vy19 - vx53 + vy53 >= 1
 vx18 - vy18 - vx53 + vy53 >= 1
 vx18 - vy18 - vx54 + vy54 >= 1
 vx19 - vy19 - vx53 + vy53 >= 1
 vx19 - vy19 - vx54 + vy54 >= 1
 vx18 - vy18 - vx54 + vy54 >= 1
 vx18 - vy18 - vx55 + vy55 >= 1
 vx19 - vy19 - vx54 + vy54 >= 1
 vx19 - vy19 - vx55 + vy55 >= 1
 vx18 - vy18 - vx55 + vy55 >= 1
 vx18 - vy18 - vx56 + vy56 >= 1
 vx19 - vy19 - vx55 + vy55 >= 1
 vx19 - vy19 - vx56 + vy56 >= 1
 vx18 - vy18 - vx56 + vy56 >= 1
 vx18 - vy18 - vx57 + vy57 >= 1
 vx19 - vy19 - vx56 + vy56 >= 1
 vx19 - vy19 - vx57 + vy57 >= 1
 vx18 - vy18 - vx57 + vy57 >= 1
 vx18 - vy18 - vx58 + vy58 >= 1
 vx19 - vy19 - vx57 + vy57 >= 1
 vx19 - vy19 - vx58 + vy58 >= 1
 vx18 - vy18 - vx58 + vy58 >= 1
 vx18 - vy18 - vx59 + vy59 >= 1
 vx19 - vy19 - vx58 + vy58 >= 1
 vx19 - vy19 - vx59 + vy59 >= 1
 vx18 - vy18 - vx59 + vy59 >= 1
 vx18 - vy18 - vx60 + vy60 >= 1
 vx19 - vy19 - vx59 + vy59 >= 1
 vx19 - vy19 - vx60 + vy60 >= 1
 vx18 - vy18 - vx60 + vy60 >= 1
 vx18 - vy18 - vx61 + vy61 >= 1
 vx19 - vy19 - vx60 + vy60 >= 1
 vx19 - vy19 - vx61 + vy61 >= 1
 q32 - oa32 - ob32 - oc32 - od32 = 0
 a33 + a34 - 2 ua32 - oa32 = 0
 b33 + b34 - 2 ub32 - ob32 = 0
 c33 + c34 - 2 uc32 - oc32 = 0
 d33 + d34 - 2 ud32 - od32 = 0
 vx19 + vy19 - vx21 - vy21 <= -1
 vx19 + vy19 - vx24 - vy24 <= -1
 vx26 + vy26 - vx21 - vy21 <= -1
 vx26 + vy26 - vx24 - vy24 <= -1
 vx19 + vy19 - vx24 - vy24 <= -1
 vx19 + vy19 - vx23 - vy23 <= -1
 vx26 + vy26 - vx24 - vy24 <= -1
 vx26 + vy26 - vx23 - vy23 <= -1
 vx19 + vy19 - vx23 - vy23 <= -1
 vx19 + vy19 - vx20 - vy20 <= -1
 vx26 + vy26 - vx23 - vy23 <= -1
 vx26 + vy26 - vx20 - vy20 <= -1
 vx19 + vy19 - vx20 - vy20 <= -1
 vx19 + vy19 - vx25 - vy25 <= -1
 vx26 + vy26 - vx20 - vy20 <= -1
 vx26 + vy26 - vx25 - vy25 <= -1
 vx19 + vy19 - vx25 - vy25 <= -1
 vx19 + vy19 - vx22 - vy22 <= -1
 vx26 + vy26 - vx25 - vy25 <= -1
 vx26 + vy26 - vx22 - vy22 <= -1
 vx19 - vy19 - vx41 + vy41 <= -1
 vx19 - vy19 - vx64 + vy64 <= -1
 vx26 - vy26 - vx41 + vy41 <= -1
 vx26 - vy26 - vx64 + vy64 <= -1
 vx19 - vy19 - vx64 + vy64 <= -1
 vx19 - vy19 - vx43 + vy43 <= -1
 vx26 - vy26 - vx64 + vy64 <= -1
 vx26 - vy26 - vx43 + vy43 <= -1
 vx19 - vy19 - vx43 + vy43 <= -1
 vx19 - vy19 - vx44 + vy44 <= -1
 vx26 - vy26 - vx43 + vy43 <= -1
 vx26 - vy26 - vx44 + vy44 <= -1
 vx19 - vy19 - vx44 + vy44 <= -1
 vx19 - vy19 - vx3 + vy3 <= -1
 vx26 - vy26 - vx44 + vy44 <= -1
 vx26 - vy26 - vx3 + vy3 <= -1
 vx19 - vy19 - vx3 + vy3 <= -1
 vx19 - vy19 - vx45 + vy45 <= -1
 vx26 - vy26 - vx3 + vy3 <= -1
 vx26 - vy26 - vx45 + vy45 <= -1
 vx19 - vy19 - vx45 + vy45 <= -1
 vx19 - vy19 - vx46 + vy46 <= -1
 vx26 - vy26 - vx45 + vy45 <= -1
 vx26 - vy26 - vx46 + vy46 <= -1
 vx19 - vy19 - vx46 + vy46 <= -1
 vx19 - vy19 - vx47 + vy47 <= -1
 vx26 - vy26 - vx46 + vy46 <= -1
 vx26 - vy26 - vx47 + vy47 <= -1
 vx19 - vy19 - vx47 + vy47 <= -1
 vx19 - vy19 - vx48 + vy48 <= -1
 vx26 - vy26 - vx47 + vy47 <= -1
 vx26 - vy26 - vx48 + vy48 <= -1
 vx19 - vy19 - vx48 + vy48 <= -1
 vx19 - vy19 - vx35 + vy35 <= -1
 vx26 - vy26 - vx48 + vy48 <= -1
 vx26 - vy26 - vx35 + vy35 <= -1
 vy19 - vy35 >= 1
 vy19 - vy49 >= 1
 vy26 - vy35 >= 1
 vy26 - vy49 >= 1
 vx19 + vy19 - vx49 - vy49 >= 1
 vx19 + vy19 - vx50 - vy50 >= 1
 vx26 + vy26 - vx49 - vy49 >= 1
 vx26 + vy26 - vx50 - vy50 >= 1
 vx19 + vy19 - vx50 - vy50 >= 1
 vx19 + vy19 - vx51 - vy51 >= 1
 vx26 + vy26 - vx50 - vy50 >= 1
 vx26 + vy26 - vx51 - vy51 >= 1
 vx19 - vx51 >= 1
 vx19 - vx52 >= 1
 vx26 - vx51 >= 1
 vx26 - vx52 >= 1
 vx19 - vy19 - vx52 + vy52 >= 1
 vx19 - vy19 - vx62 + vy62 >= 1
 vx26 - vy26 - vx52 + vy52 >= 1
 vx26 - vy26 - vx62 + vy62 >= 1
 vx19 - vy19 - vx62 + vy62 >= 1
 vx19 - vy19 - vx53 + vy53 >= 1
 vx26 - vy26 - vx62 + vy62 >= 1
 vx26 - vy26 - vx53 + vy53 >= 1
 vx19 - vy19 - vx53 + vy53 >= 1
 vx19 - vy19 - vx54 + vy54 >= 1
 vx26 - vy26 - vx53 + vy53 >= 1
 vx26 - vy26 - vx54 + vy54 >= 1
 vx19 - vy19 - vx54 + vy54 >= 1
 vx19 - vy19 - vx55 + vy55 >= 1
 vx26 - vy26 - vx54 + vy54 >= 1
 vx26 - vy26 - vx55 + vy55 >= 1
 vx19 - vy19 - vx55 + vy55 >= 1
 vx19 - vy19 - vx56 + vy56 >= 1
 vx26 - vy26 - vx55 + vy55 >= 1
 vx26 - vy26 - vx56 + vy56 >= 1
 vx19 - vy19 - vx56 + vy56 >= 1
 vx19 - vy19 - vx57 + vy57 >= 1
 vx26 - vy26 - vx56 + vy56 >= 1
 vx26 - vy26 - vx57 + vy57 >= 1
 vx19 - vy19 - vx57 + vy57 >= 1
 vx19 - vy19 - vx58 + vy58 >= 1
 vx26 - vy26 - vx57 + vy57 >= 1
 vx26 - vy26 - vx58 + vy58 >= 1
 vx19 - vy19 - vx58 + vy58 >= 1
 vx19 - vy19 - vx59 + vy59 >= 1
 vx26 - vy26 - vx58 + vy58 >= 1
 vx26 - vy26 - vx59 + vy59 >= 1
 vx19 - vy19 - vx59 + vy59 >= 1
 vx19 - vy19 - vx60 + vy60 >= 1
 vx26 - vy26 - vx59 + vy59 >= 1
 vx26 - vy26 - vx60 + vy60 >= 1
 vx19 - vy19 - vx60 + vy60 >= 1
 vx19 - vy19 - vx61 + vy61 >= 1
 vx26 - vy26 - vx60 + vy60 >= 1
 vx26 - vy26 - vx61 + vy61 >= 1
 q33 - oa33 - ob33 - oc33 - od33 = 0
 a34 + a35 - 2 ua33 - oa33 = 0
 b34 + b35 - 2 ub33 - ob33 = 0
 c34 + c35 - 2 uc33 - oc33 = 0
 d34 + d35 - 2 ud33 - od33 = 0
 vx26 + vy26 - vx24 - vy24 <= -1
 vx26 + vy26 - vx23 - vy23 <= -1
 vx21 + vy21 - vx24 - vy24 <= -1
 vx21 + vy21 - vx23 - vy23 <= -1
 vx26 + vy26 - vx23 - vy23 <= -1
 vx26 + vy26 - vx20 - vy20 <= -1
 vx21 + vy21 - vx23 - vy23 <= -1
 vx21 + vy21 - vx20 - vy20 <= -1
 vx26 + vy26 - vx20 - vy20 <= -1
 vx26 + vy26 - vx25 - vy25 <= -1
 vx21 + vy21 - vx20 - vy20 <= -1
 vx21 + vy21 - vx25 - vy25 <= -1
 vx26 + vy26 - vx25 - vy25 <= -1
 vx26 + vy26 - vx22 - vy22 <= -1
 vx21 + vy21 - vx25 - vy25 <= -1
 vx21 + vy21 - vx22 - vy22 <= -1
 vx26 - vy26 - vx41 + vy41 <= -1
 vx26 - vy26 - vx64 + vy64 <= -1
 vx21 - vy21 - vx41 + vy41 <= -1
 vx21 - vy21 - vx64 + vy64 <= -1
 vx26 - vy26 - vx64 + vy64 <= -1
 vx26 - vy26 - vx43 + vy43 <= -1
 vx21 - vy21 - vx64 + vy64 <= -1
 vx21 - vy21 - vx43 + vy43 <= -1
 vx26 - vy26 - vx43 + vy43 <= -1
 vx26 - vy26 - vx44 + vy44 <= -1
 vx21 - vy21 - vx43 + vy43 <= -1
 vx21 - vy21 - vx44 + vy44 <= -1
 vx26 - vy26 - vx44 + vy44 <= -1
 vx26 - vy26 - vx3 + vy3 <= -1
 vx21 - vy21 - vx44 + vy44 <= -1
 vx21 - vy21 - vx3 + vy3 <= -1
 vx26 - vy26 - vx3 + vy3 <= -1
 vx26 - vy26 - vx45 + vy45 <= -1
 vx21 - vy21 - vx3 + vy3 <= -1
 vx21 - vy21 - vx45 + vy45 <= -1
 vx26 - vy26 - vx45 + vy45 <= -1
 vx26 - vy26 - vx46 + vy46 <= -1
 vx21 - vy21 - vx45 + vy45 <= -1
 vx21 - vy21 - vx46 + vy46 <= -1
 vx26 - vy26 - vx46 + vy46 <= -1
 vx26 - vy26 - vx47 + vy47 <= -1
 vx21 - vy21 - vx46 + vy46 <= -1
 vx21 - vy21 - vx47 + vy47 <= -1
 vx26 - vy26 - vx47 + vy47 <= -1
 vx26 - vy26 - vx48 + vy48 <= -1
 vx21 - vy21 - vx47 + vy47 <= -1
 vx21 - vy21 - vx48 + vy48 <= -1
 vx26 - vy26 - vx48 + vy48 <= -1
 vx26 - vy26 - vx35 + vy35 <= -1
 vx21 - vy21 - vx48 + vy48 <= -1
 vx21 - vy21 - vx35 + vy35 <= -1
 vy26 - vy35 >= 1
 vy26 - vy49 >= 1
 vy21 - vy35 >= 1
 vy21 - vy49 >= 1
 vx26 + vy26 - vx49 - vy49 >= 1
 vx26 + vy26 - vx50 - vy50 >= 1
 vx21 + vy21 - vx49 - vy49 >= 1
 vx21 + vy21 - vx50 - vy50 >= 1
 vx26 + vy26 - vx50 - vy50 >= 1
 vx26 + vy26 - vx51 - vy51 >= 1
 vx21 + vy21 - vx50 - vy50 >= 1
 vx21 + vy21 - vx51 - vy51 >= 1
 vx26 + vy26 - vx51 - vy51 >= 1
 vx26 + vy26 - vx52 - vy52 >= 1
 vx21 + vy21 - vx51 - vy51 >= 1
 vx21 + vy21 - vx52 - vy52 >= 1
 vx26 - vx52 >= 1
 vx26 - vx62 >= 1
 vx21 - vx52 >= 1
 vx21 - vx62 >= 1
 vx26 - vx62 >= 1
 vx26 - vx53 >= 1
 vx21 - vx62 >= 1
 vx21 - vx53 >= 1
 vx26 - vy26 - vx53 + vy53 >= 1
 vx26 - vy26 - vx54 + vy54 >= 1
 vx21 - vy21 - vx53 + vy53 >= 1
 vx21 - vy21 - vx54 + vy54 >= 1
 vx26 - vy26 - vx54 + vy54 >= 1
 vx26 - vy26 - vx55 + vy55 >= 1
 vx21 - vy21 - vx54 + vy54 >= 1
 vx21 - vy21 - vx55 + vy55 >= 1
 vx26 - vy26 - vx55 + vy55 >= 1
 vx26 - vy26 - vx56 + vy56 >= 1
 vx21 - vy21 - vx55 + vy55 >= 1
 vx21 - vy21 - vx56 + vy56 >= 1
 vx26 - vy26 - vx56 + vy56 >= 1
 vx26 - vy26 - vx57 + vy57 >= 1
 vx21 - vy21 - vx56 + vy56 >= 1
 vx21 - vy21 - vx57 + vy57 >= 1
 vx26 - vy26 - vx57 + vy57 >= 1
 vx26 - vy26 - vx58 + vy58 >= 1
 vx21 - vy21 - vx57 + vy57 >= 1
 vx21 - vy21 - vx58 + vy58 >= 1
 vx26 - vy26 - vx58 + vy58 >= 1
 vx26 - vy26 - vx59 + vy59 >= 1
 vx21 - vy21 - vx58 + vy58 >= 1
 vx21 - vy21 - vx59 + vy59 >= 1
 vx26 - vy26 - vx59 + vy59 >= 1
 vx26 - vy26 - vx60 + vy60 >= 1
 vx21 - vy21 - vx59 + vy59 >= 1
 vx21 - vy21 - vx60 + vy60 >= 1
 vx26 - vy26 - vx60 + vy60 >= 1
 vx26 - vy26 - vx61 + vy61 >= 1
 vx21 - vy21 - vx60 + vy60 >= 1
 vx21 - vy21 - vx61 + vy61 >= 1
 q34 - oa34 - ob34 - oc34 - od34 = 0
 a35 + a36 - 2 ua34 - oa34 = 0
 b35 + b36 - 2 ub34 - ob34 = 0
 c35 + c36 - 2 uc34 - oc34 = 0
 d35 + d36 - 2 ud34 - od34 = 0
 vx21 + vy21 - vx23 - vy23 <= -1
 vx21 + vy21 - vx20 - vy20 <= -1
 vx24 + vy24 - vx23 - vy23 <= -1
 vx24 + vy24 - vx20 - vy20 <= -1
 vx21 + vy21 - vx20 - vy20 <= -1
 vx21 + vy21 - vx25 - vy25 <= -1
 vx24 + vy24 - vx20 - vy20 <= -1
 vx24 + vy24 - vx25 - vy25 <= -1
 vx21 + vy21 - vx25 - vy25 <= -1
 vx21 + vy21 - vx22 - vy22 <= -1
 vx24 + vy24 - vx25 - vy25 <= -1
 vx24 + vy24 - vx22 - vy22 <= -1
 vx21 - vy21 - vx41 + vy41 <= -1
 vx21 - vy21 - vx64 + vy64 <= -1
 vx24 - vy24 - vx41 + vy41 <= -1
 vx24 - vy24 - vx64 + vy64 <= -1
 vx21 - vy21 - vx64 + vy64 <= -1
 vx21 - vy21 - vx43 + vy43 <= -1
 vx24 - vy24 - vx64 + vy64 <= -1
 vx24 - vy24 - vx43 + vy43 <= -1
 vx21 - vy21 - vx43 + vy43 <= -1
 vx21 - vy21 - vx44 + vy44 <= -1
 vx24 - vy24 - vx43 + vy43 <= -1
 vx24 - vy24 - vx44 + vy44 <= -1
 vx21 - vy21 - vx44 + vy44 <= -1
 vx21 - vy21 - vx3 + vy3 <= -1
 vx24 - vy24 - vx44 + vy44 <= -1
 vx24 - vy24 - vx3 + vy3 <= -1
 vx21 - vy21 - vx3 + vy3 <= -1
 vx21 - vy21 - vx45 + vy45 <= -1
 vx24 - vy24 - vx3 + vy3 <= -1
 vx24 - vy24 - vx45 + vy45 <= -1
 vx21 - vy21 - vx45 + vy45 <= -1
 vx21 - vy21 - vx46 + vy46 <= -1
 vx24 - vy24 - vx45 + vy45 <= -1
 vx24 - vy24 - vx46 + vy46 <= -1
 vx21 - vy21 - vx46 + vy46 <= -1
 vx21 - vy21 - vx47 + vy47 <= -1
 vx24 - vy24 - vx46 + vy46 <= -1
 vx24 - vy24 - vx47 + vy47 <= -1
 vx21 - vy21 - vx47 + vy47 <= -1
 vx21 - vy21 - vx48 + vy48 <= -1
 vx24 - vy24 - vx47 + vy47 <= -1
 vx24 - vy24 - vx48 + vy48 <= -1
 vx21 - vy21 - vx48 + vy48 <= -1
 vx21 - vy21 - vx35 + vy35 <= -1
 vx24 - vy24 - vx48 + vy48 <= -1
 vx24 - vy24 - vx35 + vy35 <= -1
 vy21 - vy35 >= 1
 vy21 - vy49 >= 1
 vy24 - vy35 >= 1
 vy24 - vy49 >= 1
 vx21 + vy21 - vx49 - vy49 >= 1
 vx21 + vy21 - vx50 - vy50 >= 1
 vx24 + vy24 - vx49 - vy49 >= 1
 vx24 + vy24 - vx50 - vy50 >= 1
 vx21 + vy21 - vx50 - vy50 >= 1
 vx21 + vy21 - vx51 - vy51 >= 1
 vx24 + vy24 - vx50 - vy50 >= 1
 vx24 + vy24 - vx51 - vy51 >= 1
 vx21 + vy21 - vx51 - vy51 >= 1
 vx21 + vy21 - vx52 - vy52 >= 1
 vx24 + vy24 - vx51 - vy51 >= 1
 vx24 + vy24 - vx52 - vy52 >= 1
 vx21 + vy21 - vx52 - vy52 >= 1
 vx21 + vy21 - vx62 - vy62 >= 1
 vx24 + vy24 - vx52 - vy52 >= 1
 vx24 + vy24 - vx62 - vy62 >= 1
 vx21 - vx62 >= 1
 vx21 - vx53 >= 1
 vx24 - vx62 >= 1
 vx24 - vx53 >= 1
 vx21 - vx53 >= 1
 vx21 - vx54 >= 1
 vx24 - vx53 >= 1
 vx24 - vx54 >= 1
 vx21 - vy21 - vx54 + vy54 >= 1
 vx21 - vy21 - vx55 + vy55 >= 1
 vx24 - vy24 - vx54 + vy54 >= 1
 vx24 - vy24 - vx55 + vy55 >= 1
 vx21 - vy21 - vx55 + vy55 >= 1
 vx21 - vy21 - vx56 + vy56 >= 1
 vx24 - vy24 - vx55 + vy55 >= 1
 vx24 - vy24 - vx56 + vy56 >= 1
 vx21 - vy21 - vx56 + vy56 >= 1
 vx21 - vy21 - vx57 + vy57 >= 1
 vx24 - vy24 - vx56 + vy56 >= 1
 vx24 - vy24 - vx57 + vy57 >= 1
 vx21 - vy21 - vx57 + vy57 >= 1
 vx21 - vy21 - vx58 + vy58 >= 1
 vx24 - vy24 - vx57 + vy57 >= 1
 vx24 - vy24 - vx58 + vy58 >= 1
 vx21 - vy21 - vx58 + vy58 >= 1
 vx21 - vy21 - vx59 + vy59 >= 1
 vx24 - vy24 - vx58 + vy58 >= 1
 vx24 - vy24 - vx59 + vy59 >= 1
 vx21 - vy21 - vx59 + vy59 >= 1
 vx21 - vy21 - vx60 + vy60 >= 1
 vx24 - vy24 - vx59 + vy59 >= 1
 vx24 - vy24 - vx60 + vy60 >= 1
 vx21 - vy21 - vx60 + vy60 >= 1
 vx21 - vy21 - vx61 + vy61 >= 1
 vx24 - vy24 - vx60 + vy60 >= 1
 vx24 - vy24 - vx61 + vy61 >= 1
 q35 - oa35 - ob35 - oc35 - od35 = 0
 a36 + a37 - 2 ua35 - oa35 = 0
 b36 + b37 - 2 ub35 - ob35 = 0
 c36 + c37 - 2 uc35 - oc35 = 0
 d36 + d37 - 2 ud35 - od35 = 0
 vx24 + vy24 - vx20 - vy20 <= -1
 vx24 + vy24 - vx25 - vy25 <= -1
 vx23 + vy23 - vx20 - vy20 <= -1
 vx23 + vy23 - vx25 - vy25 <= -1
 vx24 + vy24 - vx25 - vy25 <= -1
 vx24 + vy24 - vx22 - vy22 <= -1
 vx23 + vy23 - vx25 - vy25 <= -1
 vx23 + vy23 - vx22 - vy22 <= -1
 vx24 - vy24 - vx41 + vy41 <= -1
 vx24 - vy24 - vx64 + vy64 <= -1
 vx23 - vy23 - vx41 + vy41 <= -1
 vx23 - vy23 - vx64 + vy64 <= -1
 vx24 - vy24 - vx64 + vy64 <= -1
 vx24 - vy24 - vx43 + vy43 <= -1
 vx23 - vy23 - vx64 + vy64 <= -1
 vx23 - vy23 - vx43 + vy43 <= -1
 vx24 - vy24 - vx43 + vy43 <= -1
 vx24 - vy24 - vx44 + vy44 <= -1
 vx23 - vy23 - vx43 + vy43 <= -1
 vx23 - vy23 - vx44 + vy44 <= -1
 vx24 - vy24 - vx44 + vy44 <= -1
 vx24 - vy24 - vx3 + vy3 <= -1
 vx23 - vy23 - vx44 + vy44 <= -1
 vx23 - vy23 - vx3 + vy3 <= -1
 vx24 - vy24 - vx3 + vy3 <= -1
 vx24 - vy24 - vx45 + vy45 <= -1
 vx23 - vy23 - vx3 + vy3 <= -1
 vx23 - vy23 - vx45 + vy45 <= -1
 vx24 - vy24 - vx45 + vy45 <= -1
 vx24 - vy24 - vx46 + vy46 <= -1
 vx23 - vy23 - vx45 + vy45 <= -1
 vx23 - vy23 - vx46 + vy46 <= -1
 vx24 - vy24 - vx46 + vy46 <= -1
 vx24 - vy24 - vx47 + vy47 <= -1
 vx23 - vy23 - vx46 + vy46 <= -1
 vx23 - vy23 - vx47 + vy47 <= -1
 vx24 - vy24 - vx47 + vy47 <= -1
 vx24 - vy24 - vx48 + vy48 <= -1
 vx23 - vy23 - vx47 + vy47 <= -1
 vx23 - vy23 - vx48 + vy48 <= -1
 vx24 - vy24 - vx48 + vy48 <= -1
 vx24 - vy24 - vx35 + vy35 <= -1
 vx23 - vy23 - vx48 + vy48 <= -1
 vx23 - vy23 - vx35 + vy35 <= -1
 vy24 - vy35 >= 1
 vy24 - vy49 >= 1
 vy23 - vy35 >= 1
 vy23 - vy49 >= 1
 vx24 + vy24 - vx49 - vy49 >= 1
 vx24 + vy24 - vx50 - vy50 >= 1
 vx23 + vy23 - vx49 - vy49 >= 1
 vx23 + vy23 - vx50 - vy50 >= 1
 vx24 + vy24 - vx50 - vy50 >= 1
 vx24 + vy24 - vx51 - vy51 >= 1
 vx23 + vy23 - vx50 - vy50 >= 1
 vx23 + vy23 - vx51 - vy51 >= 1
 vx24 + vy24 - vx51 - vy51 >= 1
 vx24 + vy24 - vx52 - vy52 >= 1
 vx23 + vy23 - vx51 - vy51 >= 1
 vx23 + vy23 - vx52 - vy52 >= 1
 vx24 + vy24 - vx52 - vy52 >= 1
 vx24 + vy24 - vx62 - vy62 >= 1
 vx23 + vy23 - vx52 - vy52 >= 1
 vx23 + vy23 - vx62 - vy62 >= 1
 vx24 + vy24 - vx62 - vy62 >= 1
 vx24 + vy24 - vx53 - vy53 >= 1
 vx23 + vy23 - vx62 - vy62 >= 1
 vx23 + vy23 - vx53 - vy53 >= 1
 vx24 - vx53 >= 1
 vx24 - vx54 >= 1
 vx23 - vx53 >= 1
 vx23 - vx54 >= 1
 vx24 - vx54 >= 1
 vx24 - vx55 >= 1
 vx23 - vx54 >= 1
 vx23 - vx55 >= 1
 vx24 - vy24 - vx55 + vy55 >= 1
 vx24 - vy24 - vx56 + vy56 >= 1
 vx23 - vy23 - vx55 + vy55 >= 1
 vx23 - vy23 - vx56 + vy56 >= 1
 vx24 - vy24 - vx56 + vy56 >= 1
 vx24 - vy24 - vx57 + vy57 >= 1
 vx23 - vy23 - vx56 + vy56 >= 1
 vx23 - vy23 - vx57 + vy57 >= 1
 vx24 - vy24 - vx57 + vy57 >= 1
 vx24 - vy24 - vx58 + vy58 >= 1
 vx23 - vy23 - vx57 + vy57 >= 1
 vx23 - vy23 - vx58 + vy58 >= 1
 vx24 - vy24 - vx58 + vy58 >= 1
 vx24 - vy24 - vx59 + vy59 >= 1
 vx23 - vy23 - vx58 + vy58 >= 1
 vx23 - vy23 - vx59 + vy59 >= 1
 vx24 - vy24 - vx59 + vy59 >= 1
 vx24 - vy24 - vx60 + vy60 >= 1
 vx23 - vy23 - vx59 + vy59 >= 1
 vx23 - vy23 - vx60 + vy60 >= 1
 vx24 - vy24 - vx60 + vy60 >= 1
 vx24 - vy24 - vx61 + vy61 >= 1
 vx23 - vy23 - vx60 + vy60 >= 1
 vx23 - vy23 - vx61 + vy61 >= 1
 q36 - oa36 - ob36 - oc36 - od36 = 0
 a37 + a38 - 2 ua36 - oa36 = 0
 b37 + b38 - 2 ub36 - ob36 = 0
 c37 + c38 - 2 uc36 - oc36 = 0
 d37 + d38 - 2 ud36 - od36 = 0
 vx23 + vy23 - vx25 - vy25 <= -1
 vx23 + vy23 - vx22 - vy22 <= -1
 vx20 + vy20 - vx25 - vy25 <= -1
 vx20 + vy20 - vx22 - vy22 <= -1
 vx23 - vy23 - vx41 + vy41 <= -1
 vx23 - vy23 - vx64 + vy64 <= -1
 vx20 - vy20 - vx41 + vy41 <= -1
 vx20 - vy20 - vx64 + vy64 <= -1
 vx23 - vy23 - vx64 + vy64 <= -1
 vx23 - vy23 - vx43 + vy43 <= -1
 vx20 - vy20 - vx64 + vy64 <= -1
 vx20 - vy20 - vx43 + vy43 <= -1
 vx23 - vy23 - vx43 + vy43 <= -1
 vx23 - vy23 - vx44 + vy44 <= -1
 vx20 - vy20 - vx43 + vy43 <= -1
 vx20 - vy20 - vx44 + vy44 <= -1
 vx23 - vy23 - vx44 + vy44 <= -1
 vx23 - vy23 - vx3 + vy3 <= -1
 vx20 - vy20 - vx44 + vy44 <= -1
 vx20 - vy20 - vx3 + vy3 <= -1
 vx23 - vy23 - vx3 + vy3 <= -1
 vx23 - vy23 - vx45 + vy45 <= -1
 vx20 - vy20 - vx3 + vy3 <= -1
 vx20 - vy20 - vx45 + vy45 <= -1
 vx23 - vy23 - vx45 + vy45 <= -1
 vx23 - vy23 - vx46 + vy46 <= -1
 vx20 - vy20 - vx45 + vy45 <= -1
 vx20 - vy20 - vx46 + vy46 <= -1
 vx23 - vy23 - vx46 + vy46 <= -1
 vx23 - vy23 - vx47 + vy47 <= -1
 vx20 - vy20 - vx46 + vy46 <= -1
 vx20 - vy20 - vx47 + vy47 <= -1
 vx23 - vy23 - vx47 + vy47 <= -1
 vx23 - vy23 - vx48 + vy48 <= -1
 vx20 - vy20 - vx47 + vy47 <= -1
 vx20 - vy20 - vx48 + vy48 <= -1
 vx23 - vy23 - vx48 + vy48 <= -1
 vx23 - vy23 - vx35 + vy35 <= -1
 vx20 - vy20 - vx48 + vy48 <= -1
 vx20 - vy20 - vx35 + vy35 <= -1
 vy23 - vy35 >= 1
 vy23 - vy49 >= 1
 vy20 - vy35 >= 1
 vy20 - vy49 >= 1
 vx23 + vy23 - vx49 - vy49 >= 1
 vx23 + vy23 - vx50 - vy50 >= 1
 vx20 + vy20 - vx49 - vy49 >= 1
 vx20 + vy20 - vx50 - vy50 >= 1
 vx23 + vy23 - vx50 - vy50 >= 1
 vx23 + vy23 - vx51 - vy51 >= 1
 vx20 + vy20 - vx50 - vy50 >= 1
 vx20 + vy20 - vx51 - vy51 >= 1
 vx23 + vy23 - vx51 - vy51 >= 1
 vx23 + vy23 - vx52 - vy52 >= 1
 vx20 + vy20 - vx51 - vy51 >= 1
 vx20 + vy20 - vx52 - vy52 >= 1
 vx23 + vy23 - vx52 - vy52 >= 1
 vx23 + vy23 - vx62 - vy62 >= 1
 vx20 + vy20 - vx52 - vy52 >= 1
 vx20 + vy20 - vx62 - vy62 >= 1
 vx23 + vy23 - vx62 - vy62 >= 1
 vx23 + vy23 - vx53 - vy53 >= 1
 vx20 + vy20 - vx62 - vy62 >= 1
 vx20 + vy20 - vx53 - vy53 >= 1
 vx23 + vy23 - vx53 - vy53 >= 1
 vx23 + vy23 - vx54 - vy54 >= 1
 vx20 + vy20 - vx53 - vy53 >= 1
 vx20 + vy20 - vx54 - vy54 >= 1
 vx23 - vx54 >= 1
 vx23 - vx55 >= 1
 vx20 - vx54 >= 1
 vx20 - vx55 >= 1
 vx23 - vy23 - vx55 + vy55 >= 1
 vx23 - vy23 - vx56 + vy56 >= 1
 vx20 - vy20 - vx55 + vy55 >= 1
 vx20 - vy20 - vx56 + vy56 >= 1
 vx23 - vy23 - vx56 + vy56 >= 1
 vx23 - vy23 - vx57 + vy57 >= 1
 vx20 - vy20 - vx56 + vy56 >= 1
 vx20 - vy20 - vx57 + vy57 >= 1
 vx23 - vy23 - vx57 + vy57 >= 1
 vx23 - vy23 - vx58 + vy58 >= 1
 vx20 - vy20 - vx57 + vy57 >= 1
 vx20 - vy20 - vx58 + vy58 >= 1
 vx23 - vy23 - vx58 + vy58 >= 1
 vx23 - vy23 - vx59 + vy59 >= 1
 vx20 - vy20 - vx58 + vy58 >= 1
 vx20 - vy20 - vx59 + vy59 >= 1
 vx23 - vy23 - vx59 + vy59 >= 1
 vx23 - vy23 - vx60 + vy60 >= 1
 vx20 - vy20 - vx59 + vy59 >= 1
 vx20 - vy20 - vx60 + vy60 >= 1
 vx23 - vy23 - vx60 + vy60 >= 1
 vx23 - vy23 - vx61 + vy61 >= 1
 vx20 - vy20 - vx60 + vy60 >= 1
 vx20 - vy20 - vx61 + vy61 >= 1
 q37 - oa37 - ob37 - oc37 - od37 = 0
 a38 + a39 - 2 ua37 - oa37 = 0
 b38 + b39 - 2 ub37 - ob37 = 0
 c38 + c39 - 2 uc37 - oc37 = 0
 d38 + d39 - 2 ud37 - od37 = 0
 vx20 - vy20 - vx41 + vy41 <= -1
 vx20 - vy20 - vx64 + vy64 <= -1
 vx25 - vy25 - vx41 + vy41 <= -1
 vx25 - vy25 - vx64 + vy64 <= -1
 vx20 - vy20 - vx64 + vy64 <= -1
 vx20 - vy20 - vx43 + vy43 <= -1
 vx25 - vy25 - vx64 + vy64 <= -1
 vx25 - vy25 - vx43 + vy43 <= -1
 vx20 - vy20 - vx43 + vy43 <= -1
 vx20 - vy20 - vx44 + vy44 <= -1
 vx25 - vy25 - vx43 + vy43 <= -1
 vx25 - vy25 - vx44 + vy44 <= -1
 vx20 - vy20 - vx44 + vy44 <= -1
 vx20 - vy20 - vx3 + vy3 <= -1
 vx25 - vy25 - vx44 + vy44 <= -1
 vx25 - vy25 - vx3 + vy3 <= -1
 vx20 - vy20 - vx3 + vy3 <= -1
 vx20 - vy20 - vx45 + vy45 <= -1
 vx25 - vy25 - vx3 + vy3 <= -1
 vx25 - vy25 - vx45 + vy45 <= -1
 vx20 - vy20 - vx45 + vy45 <= -1
 vx20 - vy20 - vx46 + vy46 <= -1
 vx25 - vy25 - vx45 + vy45 <= -1
 vx25 - vy25 - vx46 + vy46 <= -1
 vx20 - vy20 - vx46 + vy46 <= -1
 vx20 - vy20 - vx47 + vy47 <= -1
 vx25 - vy25 - vx46 + vy46 <= -1
 vx25 - vy25 - vx47 + vy47 <= -1
 vx20 - vy20 - vx47 + vy47 <= -1
 vx20 - vy20 - vx48 + vy48 <= -1
 vx25 - vy25 - vx47 + vy47 <= -1
 vx25 - vy25 - vx48 + vy48 <= -1
 vx20 - vy20 - vx48 + vy48 <= -1
 vx20 - vy20 - vx35 + vy35 <= -1
 vx25 - vy25 - vx48 + vy48 <= -1
 vx25 - vy25 - vx35 + vy35 <= -1
 vx20 + vy20 - vx35 - vy35 >= 1
 vx20 + vy20 - vx49 - vy49 >= 1
 vx25 + vy25 - vx35 - vy35 >= 1
 vx25 + vy25 - vx49 - vy49 >= 1
 vx20 + vy20 - vx49 - vy49 >= 1
 vx20 + vy20 - vx50 - vy50 >= 1
 vx25 + vy25 - vx49 - vy49 >= 1
 vx25 + vy25 - vx50 - vy50 >= 1
 vx20 + vy20 - vx50 - vy50 >= 1
 vx20 + vy20 - vx51 - vy51 >= 1
 vx25 + vy25 - vx50 - vy50 >= 1
 vx25 + vy25 - vx51 - vy51 >= 1
 vx20 + vy20 - vx51 - vy51 >= 1
 vx20 + vy20 - vx52 - vy52 >= 1
 vx25 + vy25 - vx51 - vy51 >= 1
 vx25 + vy25 - vx52 - vy52 >= 1
 vx20 + vy20 - vx52 - vy52 >= 1
 vx20 + vy20 - vx62 - vy62 >= 1
 vx25 + vy25 - vx52 - vy52 >= 1
 vx25 + vy25 - vx62 - vy62 >= 1
 vx20 + vy20 - vx62 - vy62 >= 1
 vx20 + vy20 - vx53 - vy53 >= 1
 vx25 + vy25 - vx62 - vy62 >= 1
 vx25 + vy25 - vx53 - vy53 >= 1
 vx20 + vy20 - vx53 - vy53 >= 1
 vx20 + vy20 - vx54 - vy54 >= 1
 vx25 + vy25 - vx53 - vy53 >= 1
 vx25 + vy25 - vx54 - vy54 >= 1
 vx20 + vy20 - vx54 - vy54 >= 1
 vx20 + vy20 - vx55 - vy55 >= 1
 vx25 + vy25 - vx54 - vy54 >= 1
 vx25 + vy25 - vx55 - vy55 >= 1
 vx20 - vx55 >= 1
 vx20 - vx56 >= 1
 vx25 - vx55 >= 1
 vx25 - vx56 >= 1
 vx20 - vx56 >= 1
 vx20 - vx57 >= 1
 vx25 - vx56 >= 1
 vx25 - vx57 >= 1
 vx20 - vy20 - vx57 + vy57 >= 1
 vx20 - vy20 - vx58 + vy58 >= 1
 vx25 - vy25 - vx57 + vy57 >= 1
 vx25 - vy25 - vx58 + vy58 >= 1
 vx20 - vy20 - vx58 + vy58 >= 1
 vx20 - vy20 - vx59 + vy59 >= 1
 vx25 - vy25 - vx58 + vy58 >= 1
 vx25 - vy25 - vx59 + vy59 >= 1
 vx20 - vy20 - vx59 + vy59 >= 1
 vx20 - vy20 - vx60 + vy60 >= 1
 vx25 - vy25 - vx59 + vy59 >= 1
 vx25 - vy25 - vx60 + vy60 >= 1
 vx20 - vy20 - vx60 + vy60 >= 1
 vx20 - vy20 - vx61 + vy61 >= 1
 vx25 - vy25 - vx60 + vy60 >= 1
 vx25 - vy25 - vx61 + vy61 >= 1
 vx25 - vy25 - vx41 + vy41 <= -1
 vx25 - vy25 - vx64 + vy64 <= -1
 vx22 - vy22 - vx41 + vy41 <= -1
 vx22 - vy22 - vx64 + vy64 <= -1
 vx25 - vy25 - vx64 + vy64 <= -1
 vx25 - vy25 - vx43 + vy43 <= -1
 vx22 - vy22 - vx64 + vy64 <= -1
 vx22 - vy22 - vx43 + vy43 <= -1
 vx25 - vy25 - vx43 + vy43 <= -1
 vx25 - vy25 - vx44 + vy44 <= -1
 vx22 - vy22 - vx43 + vy43 <= -1
 vx22 - vy22 - vx44 + vy44 <= -1
 vx25 - vy25 - vx44 + vy44 <= -1
 vx25 - vy25 - vx3 + vy3 <= -1
 vx22 - vy22 - vx44 + vy44 <= -1
 vx22 - vy22 - vx3 + vy3 <= -1
 vx25 - vy25 - vx3 + vy3 <= -1
 vx25 - vy25 - vx45 + vy45 <= -1
 vx22 - vy22 - vx3 + vy3 <= -1
 vx22 - vy22 - vx45 + vy45 <= -1
 vx25 - vy25 - vx45 + vy45 <= -1
 vx25 - vy25 - vx46 + vy46 <= -1
 vx22 - vy22 - vx45 + vy45 <= -1
 vx22 - vy22 - vx46 + vy46 <= -1
 vx25 - vy25 - vx46 + vy46 <= -1
 vx25 - vy25 - vx47 + vy47 <= -1
 vx22 - vy22 - vx46 + vy46 <= -1
 vx22 - vy22 - vx47 + vy47 <= -1
 vx25 - vy25 - vx47 + vy47 <= -1
 vx25 - vy25 - vx48 + vy48 <= -1
 vx22 - vy22 - vx47 + vy47 <= -1
 vx22 - vy22 - vx48 + vy48 <= -1
 vy25 - vy48 >= 1
 vy25 - vy35 >= 1
 vy22 - vy48 >= 1
 vy22 - vy35 >= 1
 vx25 + vy25 - vx35 - vy35 >= 1
 vx25 + vy25 - vx49 - vy49 >= 1
 vx22 + vy22 - vx35 - vy35 >= 1
 vx22 + vy22 - vx49 - vy49 >= 1
 vx25 + vy25 - vx49 - vy49 >= 1
 vx25 + vy25 - vx50 - vy50 >= 1
 vx22 + vy22 - vx49 - vy49 >= 1
 vx22 + vy22 - vx50 - vy50 >= 1
 vx25 + vy25 - vx50 - vy50 >= 1
 vx25 + vy25 - vx51 - vy51 >= 1
 vx22 + vy22 - vx50 - vy50 >= 1
 vx22 + vy22 - vx51 - vy51 >= 1
 vx25 + vy25 - vx51 - vy51 >= 1
 vx25 + vy25 - vx52 - vy52 >= 1
 vx22 + vy22 - vx51 - vy51 >= 1
 vx22 + vy22 - vx52 - vy52 >= 1
 vx25 + vy25 - vx52 - vy52 >= 1
 vx25 + vy25 - vx62 - vy62 >= 1
 vx22 + vy22 - vx52 - vy52 >= 1
 vx22 + vy22 - vx62 - vy62 >= 1
 vx25 + vy25 - vx62 - vy62 >= 1
 vx25 + vy25 - vx53 - vy53 >= 1
 vx22 + vy22 - vx62 - vy62 >= 1
 vx22 + vy22 - vx53 - vy53 >= 1
 vx25 + vy25 - vx53 - vy53 >= 1
 vx25 + vy25 - vx54 - vy54 >= 1
 vx22 + vy22 - vx53 - vy53 >= 1
 vx22 + vy22 - vx54 - vy54 >= 1
 vx25 + vy25 - vx54 - vy54 >= 1
 vx25 + vy25 - vx55 - vy55 >= 1
 vx22 + vy22 - vx54 - vy54 >= 1
 vx22 + vy22 - vx55 - vy55 >= 1
 vx25 + vy25 - vx55 - vy55 >= 1
 vx25 + vy25 - vx56 - vy56 >= 1
 vx22 + vy22 - vx55 - vy55 >= 1
 vx22 + vy22 - vx56 - vy56 >= 1
 vx25 + vy25 - vx56 - vy56 >= 1
 vx25 + vy25 - vx57 - vy57 >= 1
 vx22 + vy22 - vx56 - vy56 >= 1
 vx22 + vy22 - vx57 - vy57 >= 1
 vx25 - vx57 >= 1
 vx25 - vx58 >= 1
 vx22 - vx57 >= 1
 vx22 - vx58 >= 1
 vx25 - vy25 - vx58 + vy58 >= 1
 vx25 - vy25 - vx59 + vy59 >= 1
 vx22 - vy22 - vx58 + vy58 >= 1
 vx22 - vy22 - vx59 + vy59 >= 1
 vx25 - vy25 - vx59 + vy59 >= 1
 vx25 - vy25 - vx60 + vy60 >= 1
 vx22 - vy22 - vx59 + vy59 >= 1
 vx22 - vy22 - vx60 + vy60 >= 1
 vx25 - vy25 - vx60 + vy60 >= 1
 vx25 - vy25 - vx61 + vy61 >= 1
 vx22 - vy22 - vx60 + vy60 >= 1
 vx22 - vy22 - vx61 + vy61 >= 1
 q38 - oa38 - ob38 - oc38 - od38 = 0
 a40 + a41 - 2 ua38 - oa38 = 0
 b40 + b41 - 2 ub38 - ob38 = 0
 c40 + c41 - 2 uc38 - oc38 = 0
 d40 + d41 - 2 ud38 - od38 = 0
 vx41 - vy41 - vx43 + vy43 >= 1
 vx41 - vy41 - vx44 + vy44 >= 1
 vx64 - vy64 - vx43 + vy43 >= 1
 vx64 - vy64 - vx44 + vy44 >= 1
 vx41 - vy41 - vx44 + vy44 >= 1
 vx41 - vy41 - vx3 + vy3 >= 1
 vx64 - vy64 - vx44 + vy44 >= 1
 vx64 - vy64 - vx3 + vy3 >= 1
 vx41 - vy41 - vx3 + vy3 >= 1
 vx41 - vy41 - vx45 + vy45 >= 1
 vx64 - vy64 - vx3 + vy3 >= 1
 vx64 - vy64 - vx45 + vy45 >= 1
 vx41 - vy41 - vx45 + vy45 >= 1
 vx41 - vy41 - vx46 + vy46 >= 1
 vx64 - vy64 - vx45 + vy45 >= 1
 vx64 - vy64 - vx46 + vy46 >= 1
 vx41 - vy41 - vx46 + vy46 >= 1
 vx41 - vy41 - vx47 + vy47 >= 1
 vx64 - vy64 - vx46 + vy46 >= 1
 vx64 - vy64 - vx47 + vy47 >= 1
 vx41 - vy41 - vx47 + vy47 >= 1
 vx41 - vy41 - vx48 + vy48 >= 1
 vx64 - vy64 - vx47 + vy47 >= 1
 vx64 - vy64 - vx48 + vy48 >= 1
 vx41 - vy41 - vx48 + vy48 >= 1
 vx41 - vy41 - vx35 + vy35 >= 1
 vx64 - vy64 - vx48 + vy48 >= 1
 vx64 - vy64 - vx35 + vy35 >= 1
 vx41 - vy41 - vx35 + vy35 >= 1
 vx41 - vy41 - vx49 + vy49 >= 1
 vx64 - vy64 - vx35 + vy35 >= 1
 vx64 - vy64 - vx49 + vy49 >= 1
 vx41 - vy41 - vx49 + vy49 >= 1
 vx41 - vy41 - vx50 + vy50 >= 1
 vx64 - vy64 - vx49 + vy49 >= 1
 vx64 - vy64 - vx50 + vy50 >= 1
 vx41 - vy41 - vx50 + vy50 >= 1
 vx41 - vy41 - vx51 + vy51 >= 1
 vx64 - vy64 - vx50 + vy50 >= 1
 vx64 - vy64 - vx51 + vy51 >= 1
 vx41 - vy41 - vx51 + vy51 >= 1
 vx41 - vy41 - vx52 + vy52 >= 1
 vx64 - vy64 - vx51 + vy51 >= 1
 vx64 - vy64 - vx52 + vy52 >= 1
 vx41 - vy41 - vx52 + vy52 >= 1
 vx41 - vy41 - vx62 + vy62 >= 1
 vx64 - vy64 - vx52 + vy52 >= 1
 vx64 - vy64 - vx62 + vy62 >= 1
 vx41 - vy41 - vx62 + vy62 >= 1
 vx41 - vy41 - vx53 + vy53 >= 1
 vx64 - vy64 - vx62 + vy62 >= 1
 vx64 - vy64 - vx53 + vy53 >= 1
 vx41 - vy41 - vx53 + vy53 >= 1
 vx41 - vy41 - vx54 + vy54 >= 1
 vx64 - vy64 - vx53 + vy53 >= 1
 vx64 - vy64 - vx54 + vy54 >= 1
 vx41 - vy41 - vx54 + vy54 >= 1
 vx41 - vy41 - vx55 + vy55 >= 1
 vx64 - vy64 - vx54 + vy54 >= 1
 vx64 - vy64 - vx55 + vy55 >= 1
 vx41 - vy41 - vx55 + vy55 >= 1
 vx41 - vy41 - vx56 + vy56 >= 1
 vx64 - vy64 - vx55 + vy55 >= 1
 vx64 - vy64 - vx56 + vy56 >= 1
 vx41 - vy41 - vx56 + vy56 >= 1
 vx41 - vy41 - vx57 + vy57 >= 1
 vx64 - vy64 - vx56 + vy56 >= 1
 vx64 - vy64 - vx57 + vy57 >= 1
 vx41 - vy41 - vx57 + vy57 >= 1
 vx41 - vy41 - vx58 + vy58 >= 1
 vx64 - vy64 - vx57 + vy57 >= 1
 vx64 - vy64 - vx58 + vy58 >= 1
 vx41 - vy41 - vx58 + vy58 >= 1
 vx41 - vy41 - vx59 + vy59 >= 1
 vx64 - vy64 - vx58 + vy58 >= 1
 vx64 - vy64 - vx59 + vy59 >= 1
 vx41 - vy41 - vx59 + vy59 >= 1
 vx41 - vy41 - vx60 + vy60 >= 1
 vx64 - vy64 - vx59 + vy59 >= 1
 vx64 - vy64 - vx60 + vy60 >= 1
 vx41 - vy41 - vx60 + vy60 >= 1
 vx41 - vy41 - vx61 + vy61 >= 1
 vx64 - vy64 - vx60 + vy60 >= 1
 vx64 - vy64 - vx61 + vy61 >= 1
 q39 - oa39 - ob39 - oc39 - od39 = 0
 a41 + a42 - 2 ua39 - oa39 = 0
 b41 + b42 - 2 ub39 - ob39 = 0
 c41 + c42 - 2 uc39 - oc39 = 0
 d41 + d42 - 2 ud39 - od39 = 0
 vx64 - vy64 - vx44 + vy44 >= 1
 vx64 - vy64 - vx3 + vy3 >= 1
 vx43 - vy43 - vx44 + vy44 >= 1
 vx43 - vy43 - vx3 + vy3 >= 1
 vx64 - vy64 - vx3 + vy3 >= 1
 vx64 - vy64 - vx45 + vy45 >= 1
 vx43 - vy43 - vx3 + vy3 >= 1
 vx43 - vy43 - vx45 + vy45 >= 1
 vx64 - vy64 - vx45 + vy45 >= 1
 vx64 - vy64 - vx46 + vy46 >= 1
 vx43 - vy43 - vx45 + vy45 >= 1
 vx43 - vy43 - vx46 + vy46 >= 1
 vx64 - vy64 - vx46 + vy46 >= 1
 vx64 - vy64 - vx47 + vy47 >= 1
 vx43 - vy43 - vx46 + vy46 >= 1
 vx43 - vy43 - vx47 + vy47 >= 1
 vx64 - vy64 - vx47 + vy47 >= 1
 vx64 - vy64 - vx48 + vy48 >= 1
 vx43 - vy43 - vx47 + vy47 >= 1
 vx43 - vy43 - vx48 + vy48 >= 1
 vx64 - vy64 - vx48 + vy48 >= 1
 vx64 - vy64 - vx35 + vy35 >= 1
 vx43 - vy43 - vx48 + vy48 >= 1
 vx43 - vy43 - vx35 + vy35 >= 1
 vx64 - vy64 - vx35 + vy35 >= 1
 vx64 - vy64 - vx49 + vy49 >= 1
 vx43 - vy43 - vx35 + vy35 >= 1
 vx43 - vy43 - vx49 + vy49 >= 1
 vx64 - vy64 - vx49 + vy49 >= 1
 vx64 - vy64 - vx50 + vy50 >= 1
 vx43 - vy43 - vx49 + vy49 >= 1
 vx43 - vy43 - vx50 + vy50 >= 1
 vx64 - vy64 - vx50 + vy50 >= 1
 vx64 - vy64 - vx51 + vy51 >= 1
 vx43 - vy43 - vx50 + vy50 >= 1
 vx43 - vy43 - vx51 + vy51 >= 1
 vx64 - vy64 - vx51 + vy51 >= 1
 vx64 - vy64 - vx52 + vy52 >= 1
 vx43 - vy43 - vx51 + vy51 >= 1
 vx43 - vy43 - vx52 + vy52 >= 1
 vx64 - vy64 - vx52 + vy52 >= 1
 vx64 - vy64 - vx62 + vy62 >= 1
 vx43 - vy43 - vx52 + vy52 >= 1
 vx43 - vy43 - vx62 + vy62 >= 1
 vx64 - vy64 - vx62 + vy62 >= 1
 vx64 - vy64 - vx53 + vy53 >= 1
 vx43 - vy43 - vx62 + vy62 >= 1
 vx43 - vy43 - vx53 + vy53 >= 1
 vx64 - vy64 - vx53 + vy53 >= 1
 vx64 - vy64 - vx54 + vy54 >= 1
 vx43 - vy43 - vx53 + vy53 >= 1
 vx43 - vy43 - vx54 + vy54 >= 1
 vx64 - vy64 - vx54 + vy54 >= 1
 vx64 - vy64 - vx55 + vy55 >= 1
 vx43 - vy43 - vx54 + vy54 >= 1
 vx43 - vy43 - vx55 + vy55 >= 1
 vx64 - vy64 - vx55 + vy55 >= 1
 vx64 - vy64 - vx56 + vy56 >= 1
 vx43 - vy43 - vx55 + vy55 >= 1
 vx43 - vy43 - vx56 + vy56 >= 1
 vx64 - vy64 - vx56 + vy56 >= 1
 vx64 - vy64 - vx57 + vy57 >= 1
 vx43 - vy43 - vx56 + vy56 >= 1
 vx43 - vy43 - vx57 + vy57 >= 1
 vx64 - vy64 - vx57 + vy57 >= 1
 vx64 - vy64 - vx58 + vy58 >= 1
 vx43 - vy43 - vx57 + vy57 >= 1
 vx43 - vy43 - vx58 + vy58 >= 1
 vx64 - vy64 - vx58 + vy58 >= 1
 vx64 - vy64 - vx59 + vy59 >= 1
 vx43 - vy43 - vx58 + vy58 >= 1
 vx43 - vy43 - vx59 + vy59 >= 1
 vx64 - vy64 - vx59 + vy59 >= 1
 vx64 - vy64 - vx60 + vy60 >= 1
 vx43 - vy43 - vx59 + vy59 >= 1
 vx43 - vy43 - vx60 + vy60 >= 1
 vx64 - vy64 - vx60 + vy60 >= 1
 vx64 - vy64 - vx61 + vy61 >= 1
 vx43 - vy43 - vx60 + vy60 >= 1
 vx43 - vy43 - vx61 + vy61 >= 1
 q40 - oa40 - ob40 - oc40 - od40 = 0
 a42 + a43 - 2 ua40 - oa40 = 0
 b42 + b43 - 2 ub40 - ob40 = 0
 c42 + c43 - 2 uc40 - oc40 = 0
 d42 + d43 - 2 ud40 - od40 = 0
 vx43 - vy43 - vx3 + vy3 >= 1
 vx43 - vy43 - vx45 + vy45 >= 1
 vx44 - vy44 - vx3 + vy3 >= 1
 vx44 - vy44 - vx45 + vy45 >= 1
 vx43 - vx45 >= 1
 vx43 - vx46 >= 1
 vx44 - vx45 >= 1
 vx44 - vx46 >= 1
 vx43 + vy43 - vx46 - vy46 >= 1
 vx43 + vy43 - vx47 - vy47 >= 1
 vx44 + vy44 - vx46 - vy46 >= 1
 vx44 + vy44 - vx47 - vy47 >= 1
 vx43 - vy43 - vx47 + vy47 >= 1
 vx43 - vy43 - vx48 + vy48 >= 1
 vx44 - vy44 - vx47 + vy47 >= 1
 vx44 - vy44 - vx48 + vy48 >= 1
 vx43 - vy43 - vx48 + vy48 >= 1
 vx43 - vy43 - vx35 + vy35 >= 1
 vx44 - vy44 - vx48 + vy48 >= 1
 vx44 - vy44 - vx35 + vy35 >= 1
 vx43 - vy43 - vx35 + vy35 >= 1
 vx43 - vy43 - vx49 + vy49 >= 1
 vx44 - vy44 - vx35 + vy35 >= 1
 vx44 - vy44 - vx49 + vy49 >= 1
 vx43 - vy43 - vx49 + vy49 >= 1
 vx43 - vy43 - vx50 + vy50 >= 1
 vx44 - vy44 - vx49 + vy49 >= 1
 vx44 - vy44 - vx50 + vy50 >= 1
 vx43 - vy43 - vx50 + vy50 >= 1
 vx43 - vy43 - vx51 + vy51 >= 1
 vx44 - vy44 - vx50 + vy50 >= 1
 vx44 - vy44 - vx51 + vy51 >= 1
 vx43 - vy43 - vx51 + vy51 >= 1
 vx43 - vy43 - vx52 + vy52 >= 1
 vx44 - vy44 - vx51 + vy51 >= 1
 vx44 - vy44 - vx52 + vy52 >= 1
 vx43 - vy43 - vx52 + vy52 >= 1
 vx43 - vy43 - vx62 + vy62 >= 1
 vx44 - vy44 - vx52 + vy52 >= 1
 vx44 - vy44 - vx62 + vy62 >= 1
 vx43 - vy43 - vx62 + vy62 >= 1
 vx43 - vy43 - vx53 + vy53 >= 1
 vx44 - vy44 - vx62 + vy62 >= 1
 vx44 - vy44 - vx53 + vy53 >= 1
 vx43 - vy43 - vx53 + vy53 >= 1
 vx43 - vy43 - vx54 + vy54 >= 1
 vx44 - vy44 - vx53 + vy53 >= 1
 vx44 - vy44 - vx54 + vy54 >= 1
 vx43 - vy43 - vx54 + vy54 >= 1
 vx43 - vy43 - vx55 + vy55 >= 1
 vx44 - vy44 - vx54 + vy54 >= 1
 vx44 - vy44 - vx55 + vy55 >= 1
 vx43 - vy43 - vx55 + vy55 >= 1
 vx43 - vy43 - vx56 + vy56 >= 1
 vx44 - vy44 - vx55 + vy55 >= 1
 vx44 - vy44 - vx56 + vy56 >= 1
 vx43 - vy43 - vx56 + vy56 >= 1
 vx43 - vy43 - vx57 + vy57 >= 1
 vx44 - vy44 - vx56 + vy56 >= 1
 vx44 - vy44 - vx57 + vy57 >= 1
 vx43 - vy43 - vx57 + vy57 >= 1
 vx43 - vy43 - vx58 + vy58 >= 1
 vx44 - vy44 - vx57 + vy57 >= 1
 vx44 - vy44 - vx58 + vy58 >= 1
 vx43 - vy43 - vx58 + vy58 >= 1
 vx43 - vy43 - vx59 + vy59 >= 1
 vx44 - vy44 - vx58 + vy58 >= 1
 vx44 - vy44 - vx59 + vy59 >= 1
 vx43 - vy43 - vx59 + vy59 >= 1
 vx43 - vy43 - vx60 + vy60 >= 1
 vx44 - vy44 - vx59 + vy59 >= 1
 vx44 - vy44 - vx60 + vy60 >= 1
 vx43 - vy43 - vx60 + vy60 >= 1
 vx43 - vy43 - vx61 + vy61 >= 1
 vx44 - vy44 - vx60 + vy60 >= 1
 vx44 - vy44 - vx61 + vy61 >= 1
 q41 - oa41 - ob41 - oc41 - od41 = 0
 a43 + a44 - 2 ua41 - oa41 = 0
 b43 + b44 - 2 ub41 - ob41 = 0
 c43 + c44 - 2 uc41 - oc41 = 0
 d43 + d44 - 2 ud41 - od41 = 0
 vx44 + vy44 - vx45 - vy45 >= 1
 vx44 + vy44 - vx46 - vy46 >= 1
 vx3 + vy3 - vx45 - vy45 >= 1
 vx3 + vy3 - vx46 - vy46 >= 1
 vx44 + vy44 - vx46 - vy46 >= 1
 vx44 + vy44 - vx47 - vy47 >= 1
 vx3 + vy3 - vx46 - vy46 >= 1
 vx3 + vy3 - vx47 - vy47 >= 1
 vx44 - vy44 - vx47 + vy47 >= 1
 vx44 - vy44 - vx48 + vy48 >= 1
 vx3 - vy3 - vx47 + vy47 >= 1
 vx3 - vy3 - vx48 + vy48 >= 1
 vx44 - vy44 - vx48 + vy48 >= 1
 vx44 - vy44 - vx35 + vy35 >= 1
 vx3 - vy3 - vx48 + vy48 >= 1
 vx3 - vy3 - vx35 + vy35 >= 1
 vx44 - vy44 - vx35 + vy35 >= 1
 vx44 - vy44 - vx49 + vy49 >= 1
 vx3 - vy3 - vx35 + vy35 >= 1
 vx3 - vy3 - vx49 + vy49 >= 1
 vx44 - vy44 - vx49 + vy49 >= 1
 vx44 - vy44 - vx50 + vy50 >= 1
 vx3 - vy3 - vx49 + vy49 >= 1
 vx3 - vy3 - vx50 + vy50 >= 1
 vx44 - vy44 - vx50 + vy50 >= 1
 vx44 - vy44 - vx51 + vy51 >= 1
 vx3 - vy3 - vx50 + vy50 >= 1
 vx3 - vy3 - vx51 + vy51 >= 1
 vx44 - vy44 - vx51 + vy51 >= 1
 vx44 - vy44 - vx52 + vy52 >= 1
 vx3 - vy3 - vx51 + vy51 >= 1
 vx3 - vy3 - vx52 + vy52 >= 1
 vx44 - vy44 - vx52 + vy52 >= 1
 vx44 - vy44 - vx62 + vy62 >= 1
 vx3 - vy3 - vx52 + vy52 >= 1
 vx3 - vy3 - vx62 + vy62 >= 1
 vx44 - vy44 - vx62 + vy62 >= 1
 vx44 - vy44 - vx53 + vy53 >= 1
 vx3 - vy3 - vx62 + vy62 >= 1
 vx3 - vy3 - vx53 + vy53 >= 1
 vx44 - vy44 - vx53 + vy53 >= 1
 vx44 - vy44 - vx54 + vy54 >= 1
 vx3 - vy3 - vx53 + vy53 >= 1
 vx3 - vy3 - vx54 + vy54 >= 1
 vx44 - vy44 - vx54 + vy54 >= 1
 vx44 - vy44 - vx55 + vy55 >= 1
 vx3 - vy3 - vx54 + vy54 >= 1
 vx3 - vy3 - vx55 + vy55 >= 1
 vx44 - vy44 - vx55 + vy55 >= 1
 vx44 - vy44 - vx56 + vy56 >= 1
 vx3 - vy3 - vx55 + vy55 >= 1
 vx3 - vy3 - vx56 + vy56 >= 1
 vx44 - vy44 - vx56 + vy56 >= 1
 vx44 - vy44 - vx57 + vy57 >= 1
 vx3 - vy3 - vx56 + vy56 >= 1
 vx3 - vy3 - vx57 + vy57 >= 1
 vx44 - vy44 - vx57 + vy57 >= 1
 vx44 - vy44 - vx58 + vy58 >= 1
 vx3 - vy3 - vx57 + vy57 >= 1
 vx3 - vy3 - vx58 + vy58 >= 1
 vx44 - vy44 - vx58 + vy58 >= 1
 vx44 - vy44 - vx59 + vy59 >= 1
 vx3 - vy3 - vx58 + vy58 >= 1
 vx3 - vy3 - vx59 + vy59 >= 1
 vx44 - vy44 - vx59 + vy59 >= 1
 vx44 - vy44 - vx60 + vy60 >= 1
 vx3 - vy3 - vx59 + vy59 >= 1
 vx3 - vy3 - vx60 + vy60 >= 1
 vx44 - vy44 - vx60 + vy60 >= 1
 vx44 - vy44 - vx61 + vy61 >= 1
 vx3 - vy3 - vx60 + vy60 >= 1
 vx3 - vy3 - vx61 + vy61 >= 1
 q42 - oa42 - ob42 - oc42 - od42 = 0
 a44 + a45 - 2 ua42 - oa42 = 0
 b44 + b45 - 2 ub42 - ob42 = 0
 c44 + c45 - 2 uc42 - oc42 = 0
 d44 + d45 - 2 ud42 - od42 = 0
 vx3 + vy3 - vx46 - vy46 >= 1
 vx3 + vy3 - vx47 - vy47 >= 1
 vx45 + vy45 - vx46 - vy46 >= 1
 vx45 + vy45 - vx47 - vy47 >= 1
 vx3 - vy3 - vx47 + vy47 >= 1
 vx3 - vy3 - vx48 + vy48 >= 1
 vx45 - vy45 - vx47 + vy47 >= 1
 vx45 - vy45 - vx48 + vy48 >= 1
 vx3 - vy3 - vx48 + vy48 >= 1
 vx3 - vy3 - vx35 + vy35 >= 1
 vx45 - vy45 - vx48 + vy48 >= 1
 vx45 - vy45 - vx35 + vy35 >= 1
 vx3 - vy3 - vx35 + vy35 >= 1
 vx3 - vy3 - vx49 + vy49 >= 1
 vx45 - vy45 - vx35 + vy35 >= 1
 vx45 - vy45 - vx49 + vy49 >= 1
 vx3 - vy3 - vx49 + vy49 >= 1
 vx3 - vy3 - vx50 + vy50 >= 1
 vx45 - vy45 - vx49 + vy49 >= 1
 vx45 - vy45 - vx50 + vy50 >= 1
 vx3 - vy3 - vx50 + vy50 >= 1
 vx3 - vy3 - vx51 + vy51 >= 1
 vx45 - vy45 - vx50 + vy50 >= 1
 vx45 - vy45 - vx51 + vy51 >= 1
 vx3 - vy3 - vx51 + vy51 >= 1
 vx3 - vy3 - vx52 + vy52 >= 1
 vx45 - vy45 - vx51 + vy51 >= 1
 vx45 - vy45 - vx52 + vy52 >= 1
 vx3 - vy3 - vx52 + vy52 >= 1
 vx3 - vy3 - vx62 + vy62 >= 1
 vx45 - vy45 - vx52 + vy52 >= 1
 vx45 - vy45 - vx62 + vy62 >= 1
 vx3 - vy3 - vx62 + vy62 >= 1
 vx3 - vy3 - vx53 + vy53 >= 1
 vx45 - vy45 - vx62 + vy62 >= 1
 vx45 - vy45 - vx53 + vy53 >= 1
 vx3 - vy3 - vx53 + vy53 >= 1
 vx3 - vy3 - vx54 + vy54 >= 1
 vx45 - vy45 - vx53 + vy53 >= 1
 vx45 - vy45 - vx54 + vy54 >= 1
 vx3 - vy3 - vx54 + vy54 >= 1
 vx3 - vy3 - vx55 + vy55 >= 1
 vx45 - vy45 - vx54 + vy54 >= 1
 vx45 - vy45 - vx55 + vy55 >= 1
 vx3 - vy3 - vx55 + vy55 >= 1
 vx3 - vy3 - vx56 + vy56 >= 1
 vx45 - vy45 - vx55 + vy55 >= 1
 vx45 - vy45 - vx56 + vy56 >= 1
 vx3 - vy3 - vx56 + vy56 >= 1
 vx3 - vy3 - vx57 + vy57 >= 1
 vx45 - vy45 - vx56 + vy56 >= 1
 vx45 - vy45 - vx57 + vy57 >= 1
 vx3 - vy3 - vx57 + vy57 >= 1
 vx3 - vy3 - vx58 + vy58 >= 1
 vx45 - vy45 - vx57 + vy57 >= 1
 vx45 - vy45 - vx58 + vy58 >= 1
 vx3 - vy3 - vx58 + vy58 >= 1
 vx3 - vy3 - vx59 + vy59 >= 1
 vx45 - vy45 - vx58 + vy58 >= 1
 vx45 - vy45 - vx59 + vy59 >= 1
 vx3 - vy3 - vx59 + vy59 >= 1
 vx3 - vy3 - vx60 + vy60 >= 1
 vx45 - vy45 - vx59 + vy59 >= 1
 vx45 - vy45 - vx60 + vy60 >= 1
 vx3 - vy3 - vx60 + vy60 >= 1
 vx3 - vy3 - vx61 + vy61 >= 1
 vx45 - vy45 - vx60 + vy60 >= 1
 vx45 - vy45 - vx61 + vy61 >= 1
 q43 - oa43 - ob43 - oc43 - od43 = 0
 a45 + a46 - 2 ua43 - oa43 = 0
 b45 + b46 - 2 ub43 - ob43 = 0
 c45 + c46 - 2 uc43 - oc43 = 0
 d45 + d46 - 2 ud43 - od43 = 0
 vx45 - vy45 - vx47 + vy47 >= 1
 vx45 - vy45 - vx48 + vy48 >= 1
 vx46 - vy46 - vx47 + vy47 >= 1
 vx46 - vy46 - vx48 + vy48 >= 1
 vx45 - vy45 - vx48 + vy48 >= 1
 vx45 - vy45 - vx35 + vy35 >= 1
 vx46 - vy46 - vx48 + vy48 >= 1
 vx46 - vy46 - vx35 + vy35 >= 1
 vx45 - vy45 - vx35 + vy35 >= 1
 vx45 - vy45 - vx49 + vy49 >= 1
 vx46 - vy46 - vx35 + vy35 >= 1
 vx46 - vy46 - vx49 + vy49 >= 1
 vx45 - vy45 - vx49 + vy49 >= 1
 vx45 - vy45 - vx50 + vy50 >= 1
 vx46 - vy46 - vx49 + vy49 >= 1
 vx46 - vy46 - vx50 + vy50 >= 1
 vx45 - vy45 - vx50 + vy50 >= 1
 vx45 - vy45 - vx51 + vy51 >= 1
 vx46 - vy46 - vx50 + vy50 >= 1
 vx46 - vy46 - vx51 + vy51 >= 1
 vx45 - vy45 - vx51 + vy51 >= 1
 vx45 - vy45 - vx52 + vy52 >= 1
 vx46 - vy46 - vx51 + vy51 >= 1
 vx46 - vy46 - vx52 + vy52 >= 1
 vx45 - vy45 - vx52 + vy52 >= 1
 vx45 - vy45 - vx62 + vy62 >= 1
 vx46 - vy46 - vx52 + vy52 >= 1
 vx46 - vy46 - vx62 + vy62 >= 1
 vx45 - vy45 - vx62 + vy62 >= 1
 vx45 - vy45 - vx53 + vy53 >= 1
 vx46 - vy46 - vx62 + vy62 >= 1
 vx46 - vy46 - vx53 + vy53 >= 1
 vx45 - vy45 - vx53 + vy53 >= 1
 vx45 - vy45 - vx54 + vy54 >= 1
 vx46 - vy46 - vx53 + vy53 >= 1
 vx46 - vy46 - vx54 + vy54 >= 1
 vx45 - vy45 - vx54 + vy54 >= 1
 vx45 - vy45 - vx55 + vy55 >= 1
 vx46 - vy46 - vx54 + vy54 >= 1
 vx46 - vy46 - vx55 + vy55 >= 1
 vx45 - vy45 - vx55 + vy55 >= 1
 vx45 - vy45 - vx56 + vy56 >= 1
 vx46 - vy46 - vx55 + vy55 >= 1
 vx46 - vy46 - vx56 + vy56 >= 1
 vx45 - vy45 - vx56 + vy56 >= 1
 vx45 - vy45 - vx57 + vy57 >= 1
 vx46 - vy46 - vx56 + vy56 >= 1
 vx46 - vy46 - vx57 + vy57 >= 1
 vx45 - vy45 - vx57 + vy57 >= 1
 vx45 - vy45 - vx58 + vy58 >= 1
 vx46 - vy46 - vx57 + vy57 >= 1
 vx46 - vy46 - vx58 + vy58 >= 1
 vx45 - vy45 - vx58 + vy58 >= 1
 vx45 - vy45 - vx59 + vy59 >= 1
 vx46 - vy46 - vx58 + vy58 >= 1
 vx46 - vy46 - vx59 + vy59 >= 1
 vx45 - vy45 - vx59 + vy59 >= 1
 vx45 - vy45 - vx60 + vy60 >= 1
 vx46 - vy46 - vx59 + vy59 >= 1
 vx46 - vy46 - vx60 + vy60 >= 1
 vx45 - vy45 - vx60 + vy60 >= 1
 vx45 - vy45 - vx61 + vy61 >= 1
 vx46 - vy46 - vx60 + vy60 >= 1
 vx46 - vy46 - vx61 + vy61 >= 1
 q44 - oa44 - ob44 - oc44 - od44 = 0
 a46 + a47 - 2 ua44 - oa44 = 0
 b46 + b47 - 2 ub44 - ob44 = 0
 c46 + c47 - 2 uc44 - oc44 = 0
 d46 + d47 - 2 ud44 - od44 = 0
 vx46 - vy46 - vx48 + vy48 >= 1
 vx46 - vy46 - vx35 + vy35 >= 1
 vx47 - vy47 - vx48 + vy48 >= 1
 vx47 - vy47 - vx35 + vy35 >= 1
 vx46 - vy46 - vx35 + vy35 >= 1
 vx46 - vy46 - vx49 + vy49 >= 1
 vx47 - vy47 - vx35 + vy35 >= 1
 vx47 - vy47 - vx49 + vy49 >= 1
 vx46 - vy46 - vx49 + vy49 >= 1
 vx46 - vy46 - vx50 + vy50 >= 1
 vx47 - vy47 - vx49 + vy49 >= 1
 vx47 - vy47 - vx50 + vy50 >= 1
 vx46 - vy46 - vx50 + vy50 >= 1
 vx46 - vy46 - vx51 + vy51 >= 1
 vx47 - vy47 - vx50 + vy50 >= 1
 vx47 - vy47 - vx51 + vy51 >= 1
 vx46 - vy46 - vx51 + vy51 >= 1
 vx46 - vy46 - vx52 + vy52 >= 1
 vx47 - vy47 - vx51 + vy51 >= 1
 vx47 - vy47 - vx52 + vy52 >= 1
 vx46 - vy46 - vx52 + vy52 >= 1
 vx46 - vy46 - vx62 + vy62 >= 1
 vx47 - vy47 - vx52 + vy52 >= 1
 vx47 - vy47 - vx62 + vy62 >= 1
 vx46 - vy46 - vx62 + vy62 >= 1
 vx46 - vy46 - vx53 + vy53 >= 1
 vx47 - vy47 - vx62 + vy62 >= 1
 vx47 - vy47 - vx53 + vy53 >= 1
 vx46 - vy46 - vx53 + vy53 >= 1
 vx46 - vy46 - vx54 + vy54 >= 1
 vx47 - vy47 - vx53 + vy53 >= 1
 vx47 - vy47 - vx54 + vy54 >= 1
 vx46 - vy46 - vx54 + vy54 >= 1
 vx46 - vy46 - vx55 + vy55 >= 1
 vx47 - vy47 - vx54 + vy54 >= 1
 vx47 - vy47 - vx55 + vy55 >= 1
 vx46 - vy46 - vx55 + vy55 >= 1
 vx46 - vy46 - vx56 + vy56 >= 1
 vx47 - vy47 - vx55 + vy55 >= 1
 vx47 - vy47 - vx56 + vy56 >= 1
 vx46 - vy46 - vx56 + vy56 >= 1
 vx46 - vy46 - vx57 + vy57 >= 1
 vx47 - vy47 - vx56 + vy56 >= 1
 vx47 - vy47 - vx57 + vy57 >= 1
 vx46 - vy46 - vx57 + vy57 >= 1
 vx46 - vy46 - vx58 + vy58 >= 1
 vx47 - vy47 - vx57 + vy57 >= 1
 vx47 - vy47 - vx58 + vy58 >= 1
 vx46 - vy46 - vx58 + vy58 >= 1
 vx46 - vy46 - vx59 + vy59 >= 1
 vx47 - vy47 - vx58 + vy58 >= 1
 vx47 - vy47 - vx59 + vy59 >= 1
 vx46 - vy46 - vx59 + vy59 >= 1
 vx46 - vy46 - vx60 + vy60 >= 1
 vx47 - vy47 - vx59 + vy59 >= 1
 vx47 - vy47 - vx60 + vy60 >= 1
 vx46 - vy46 - vx60 + vy60 >= 1
 vx46 - vy46 - vx61 + vy61 >= 1
 vx47 - vy47 - vx60 + vy60 >= 1
 vx47 - vy47 - vx61 + vy61 >= 1
 q45 - oa45 - ob45 - oc45 - od45 = 0
 a47 + a48 - 2 ua45 - oa45 = 0
 b47 + b48 - 2 ub45 - ob45 = 0
 c47 + c48 - 2 uc45 - oc45 = 0
 d47 + d48 - 2 ud45 - od45 = 0
 vx47 + vy47 - vx35 - vy35 >= 1
 vx47 + vy47 - vx49 - vy49 >= 1
 vx48 + vy48 - vx35 - vy35 >= 1
 vx48 + vy48 - vx49 - vy49 >= 1
 vx47 - vy47 - vx49 + vy49 >= 1
 vx47 - vy47 - vx50 + vy50 >= 1
 vx48 - vy48 - vx49 + vy49 >= 1
 vx48 - vy48 - vx50 + vy50 >= 1
 vx47 - vy47 - vx50 + vy50 >= 1
 vx47 - vy47 - vx51 + vy51 >= 1
 vx48 - vy48 - vx50 + vy50 >= 1
 vx48 - vy48 - vx51 + vy51 >= 1
 vx47 - vy47 - vx51 + vy51 >= 1
 vx47 - vy47 - vx52 + vy52 >= 1
 vx48 - vy48 - vx51 + vy51 >= 1
 vx48 - vy48 - vx52 + vy52 >= 1
 vx47 - vy47 - vx52 + vy52 >= 1
 vx47 - vy47 - vx62 + vy62 >= 1
 vx48 - vy48 - vx52 + vy52 >= 1
 vx48 - vy48 - vx62 + vy62 >= 1
 vx47 - vy47 - vx62 + vy62 >= 1
 vx47 - vy47 - vx53 + vy53 >= 1
 vx48 - vy48 - vx62 + vy62 >= 1
 vx48 - vy48 - vx53 + vy53 >= 1
 vx47 - vy47 - vx53 + vy53 >= 1
 vx47 - vy47 - vx54 + vy54 >= 1
 vx48 - vy48 - vx53 + vy53 >= 1
 vx48 - vy48 - vx54 + vy54 >= 1
 vx47 - vy47 - vx54 + vy54 >= 1
 vx47 - vy47 - vx55 + vy55 >= 1
 vx48 - vy48 - vx54 + vy54 >= 1
 vx48 - vy48 - vx55 + vy55 >= 1
 vx47 - vy47 - vx55 + vy55 >= 1
 vx47 - vy47 - vx56 + vy56 >= 1
 vx48 - vy48 - vx55 + vy55 >= 1
 vx48 - vy48 - vx56 + vy56 >= 1
 vx47 - vy47 - vx56 + vy56 >= 1
 vx47 - vy47 - vx57 + vy57 >= 1
 vx48 - vy48 - vx56 + vy56 >= 1
 vx48 - vy48 - vx57 + vy57 >= 1
 vx47 - vy47 - vx57 + vy57 >= 1
 vx47 - vy47 - vx58 + vy58 >= 1
 vx48 - vy48 - vx57 + vy57 >= 1
 vx48 - vy48 - vx58 + vy58 >= 1
 vx47 - vy47 - vx58 + vy58 >= 1
 vx47 - vy47 - vx59 + vy59 >= 1
 vx48 - vy48 - vx58 + vy58 >= 1
 vx48 - vy48 - vx59 + vy59 >= 1
 vx47 - vy47 - vx59 + vy59 >= 1
 vx47 - vy47 - vx60 + vy60 >= 1
 vx48 - vy48 - vx59 + vy59 >= 1
 vx48 - vy48 - vx60 + vy60 >= 1
 vx47 - vy47 - vx60 + vy60 >= 1
 vx47 - vy47 - vx61 + vy61 >= 1
 vx48 - vy48 - vx60 + vy60 >= 1
 vx48 - vy48 - vx61 + vy61 >= 1
 q46 - oa46 - ob46 - oc46 - od46 = 0
 a48 + a49 - 2 ua46 - oa46 = 0
 b48 + b49 - 2 ub46 - ob46 = 0
 c48 + c49 - 2 uc46 - oc46 = 0
 d48 + d49 - 2 ud46 - od46 = 0
 vx48 - vy48 - vx49 + vy49 >= 1
 vx48 - vy48 - vx50 + vy50 >= 1
 vx35 - vy35 - vx49 + vy49 >= 1
 vx35 - vy35 - vx50 + vy50 >= 1
 vx48 - vy48 - vx50 + vy50 >= 1
 vx48 - vy48 - vx51 + vy51 >= 1
 vx35 - vy35 - vx50 + vy50 >= 1
 vx35 - vy35 - vx51 + vy51 >= 1
 vx48 - vy48 - vx51 + vy51 >= 1
 vx48 - vy48 - vx52 + vy52 >= 1
 vx35 - vy35 - vx51 + vy51 >= 1
 vx35 - vy35 - vx52 + vy52 >= 1
 vx48 - vy48 - vx52 + vy52 >= 1
 vx48 - vy48 - vx62 + vy62 >= 1
 vx35 - vy35 - vx52 + vy52 >= 1
 vx35 - vy35 - vx62 + vy62 >= 1
 vx48 - vy48 - vx62 + vy62 >= 1
 vx48 - vy48 - vx53 + vy53 >= 1
 vx35 - vy35 - vx62 + vy62 >= 1
 vx35 - vy35 - vx53 + vy53 >= 1
 vx48 - vy48 - vx53 + vy53 >= 1
 vx48 - vy48 - vx54 + vy54 >= 1
 vx35 - vy35 - vx53 + vy53 >= 1
 vx35 - vy35 - vx54 + vy54 >= 1
 vx48 - vy48 - vx54 + vy54 >= 1
 vx48 - vy48 - vx55 + vy55 >= 1
 vx35 - vy35 - vx54 + vy54 >= 1
 vx35 - vy35 - vx55 + vy55 >= 1
 vx48 - vy48 - vx55 + vy55 >= 1
 vx48 - vy48 - vx56 + vy56 >= 1
 vx35 - vy35 - vx55 + vy55 >= 1
 vx35 - vy35 - vx56 + vy56 >= 1
 vx48 - vy48 - vx56 + vy56 >= 1
 vx48 - vy48 - vx57 + vy57 >= 1
 vx35 - vy35 - vx56 + vy56 >= 1
 vx35 - vy35 - vx57 + vy57 >= 1
 vx48 - vy48 - vx57 + vy57 >= 1
 vx48 - vy48 - vx58 + vy58 >= 1
 vx35 - vy35 - vx57 + vy57 >= 1
 vx35 - vy35 - vx58 + vy58 >= 1
 vx48 - vy48 - vx58 + vy58 >= 1
 vx48 - vy48 - vx59 + vy59 >= 1
 vx35 - vy35 - vx58 + vy58 >= 1
 vx35 - vy35 - vx59 + vy59 >= 1
 vx48 - vy48 - vx59 + vy59 >= 1
 vx48 - vy48 - vx60 + vy60 >= 1
 vx35 - vy35 - vx59 + vy59 >= 1
 vx35 - vy35 - vx60 + vy60 >= 1
 vx48 - vy48 - vx60 + vy60 >= 1
 vx48 - vy48 - vx61 + vy61 >= 1
 vx35 - vy35 - vx60 + vy60 >= 1
 vx35 - vy35 - vx61 + vy61 >= 1
 q47 - oa47 - ob47 - oc47 - od47 = 0
 a49 + a50 - 2 ua47 - oa47 = 0
 b49 + b50 - 2 ub47 - ob47 = 0
 c49 + c50 - 2 uc47 - oc47 = 0
 d49 + d50 - 2 ud47 - od47 = 0
 vx35 - vy35 - vx50 + vy50 >= 1
 vx35 - vy35 - vx51 + vy51 >= 1
 vx49 - vy49 - vx50 + vy50 >= 1
 vx49 - vy49 - vx51 + vy51 >= 1
 vx35 - vy35 - vx51 + vy51 >= 1
 vx35 - vy35 - vx52 + vy52 >= 1
 vx49 - vy49 - vx51 + vy51 >= 1
 vx49 - vy49 - vx52 + vy52 >= 1
 vx35 - vy35 - vx52 + vy52 >= 1
 vx35 - vy35 - vx62 + vy62 >= 1
 vx49 - vy49 - vx52 + vy52 >= 1
 vx49 - vy49 - vx62 + vy62 >= 1
 vx35 - vy35 - vx62 + vy62 >= 1
 vx35 - vy35 - vx53 + vy53 >= 1
 vx49 - vy49 - vx62 + vy62 >= 1
 vx49 - vy49 - vx53 + vy53 >= 1
 vx35 - vy35 - vx53 + vy53 >= 1
 vx35 - vy35 - vx54 + vy54 >= 1
 vx49 - vy49 - vx53 + vy53 >= 1
 vx49 - vy49 - vx54 + vy54 >= 1
 vx35 - vy35 - vx54 + vy54 >= 1
 vx35 - vy35 - vx55 + vy55 >= 1
 vx49 - vy49 - vx54 + vy54 >= 1
 vx49 - vy49 - vx55 + vy55 >= 1
 vx35 - vy35 - vx55 + vy55 >= 1
 vx35 - vy35 - vx56 + vy56 >= 1
 vx49 - vy49 - vx55 + vy55 >= 1
 vx49 - vy49 - vx56 + vy56 >= 1
 vx35 - vy35 - vx56 + vy56 >= 1
 vx35 - vy35 - vx57 + vy57 >= 1
 vx49 - vy49 - vx56 + vy56 >= 1
 vx49 - vy49 - vx57 + vy57 >= 1
 vx35 - vy35 - vx57 + vy57 >= 1
 vx35 - vy35 - vx58 + vy58 >= 1
 vx49 - vy49 - vx57 + vy57 >= 1
 vx49 - vy49 - vx58 + vy58 >= 1
 vx35 - vy35 - vx58 + vy58 >= 1
 vx35 - vy35 - vx59 + vy59 >= 1
 vx49 - vy49 - vx58 + vy58 >= 1
 vx49 - vy49 - vx59 + vy59 >= 1
 vx35 - vy35 - vx59 + vy59 >= 1
 vx35 - vy35 - vx60 + vy60 >= 1
 vx49 - vy49 - vx59 + vy59 >= 1
 vx49 - vy49 - vx60 + vy60 >= 1
 vx35 - vy35 - vx60 + vy60 >= 1
 vx35 - vy35 - vx61 + vy61 >= 1
 vx49 - vy49 - vx60 + vy60 >= 1
 vx49 - vy49 - vx61 + vy61 >= 1
 q48 - oa48 - ob48 - oc48 - od48 = 0
 a50 + a51 - 2 ua48 - oa48 = 0
 b50 + b51 - 2 ub48 - ob48 = 0
 c50 + c51 - 2 uc48 - oc48 = 0
 d50 + d51 - 2 ud48 - od48 = 0
 vx49 - vy49 - vx51 + vy51 >= 1
 vx49 - vy49 - vx52 + vy52 >= 1
 vx50 - vy50 - vx51 + vy51 >= 1
 vx50 - vy50 - vx52 + vy52 >= 1
 vy49 - vy52 <= -1
 vy49 - vy62 <= -1
 vy50 - vy52 <= -1
 vy50 - vy62 <= -1
 vy49 - vy62 <= -1
 vy49 - vy53 <= -1
 vy50 - vy62 <= -1
 vy50 - vy53 <= -1
 vy49 - vy53 <= -1
 vy49 - vy54 <= -1
 vy50 - vy53 <= -1
 vy50 - vy54 <= -1
 vy49 - vy54 <= -1
 vy49 - vy55 <= -1
 vy50 - vy54 <= -1
 vy50 - vy55 <= -1
 vy49 - vy55 <= -1
 vy49 - vy56 <= -1
 vy50 - vy55 <= -1
 vy50 - vy56 <= -1
 vy49 - vy56 <= -1
 vy49 - vy57 <= -1
 vy50 - vy56 <= -1
 vy50 - vy57 <= -1
 vy49 - vy57 <= -1
 vy49 - vy58 <= -1
 vy50 - vy57 <= -1
 vy50 - vy58 <= -1
 vy49 - vy58 <= -1
 vy49 - vy59 <= -1
 vy50 - vy58 <= -1
 vy50 - vy59 <= -1
 vx49 - vy49 - vx59 + vy59 >= 1
 vx49 - vy49 - vx60 + vy60 >= 1
 vx50 - vy50 - vx59 + vy59 >= 1
 vx50 - vy50 - vx60 + vy60 >= 1
 vx49 - vy49 - vx60 + vy60 >= 1
 vx49 - vy49 - vx61 + vy61 >= 1
 vx50 - vy50 - vx60 + vy60 >= 1
 vx50 - vy50 - vx61 + vy61 >= 1
 q49 - oa49 - ob49 - oc49 - od49 = 0
 a51 + a52 - 2 ua49 - oa49 = 0
 b51 + b52 - 2 ub49 - ob49 = 0
 c51 + c52 - 2 uc49 - oc49 = 0
 d51 + d52 - 2 ud49 - od49 = 0
 vx50 + vy50 - vx52 - vy52 <= -1
 vx50 + vy50 - vx62 - vy62 <= -1
 vx51 + vy51 - vx52 - vy52 <= -1
 vx51 + vy51 - vx62 - vy62 <= -1
 vx50 + vy50 - vx62 - vy62 <= -1
 vx50 + vy50 - vx53 - vy53 <= -1
 vx51 + vy51 - vx62 - vy62 <= -1
 vx51 + vy51 - vx53 - vy53 <= -1
 vx50 + vy50 - vx53 - vy53 <= -1
 vx50 + vy50 - vx54 - vy54 <= -1
 vx51 + vy51 - vx53 - vy53 <= -1
 vx51 + vy51 - vx54 - vy54 <= -1
 vx50 + vy50 - vx54 - vy54 <= -1
 vx50 + vy50 - vx55 - vy55 <= -1
 vx51 + vy51 - vx54 - vy54 <= -1
 vx51 + vy51 - vx55 - vy55 <= -1
 vx50 + vy50 - vx55 - vy55 <= -1
 vx50 + vy50 - vx56 - vy56 <= -1
 vx51 + vy51 - vx55 - vy55 <= -1
 vx51 + vy51 - vx56 - vy56 <= -1
 vx50 + vy50 - vx56 - vy56 <= -1
 vx50 + vy50 - vx57 - vy57 <= -1
 vx51 + vy51 - vx56 - vy56 <= -1
 vx51 + vy51 - vx57 - vy57 <= -1
 vx50 + vy50 - vx57 - vy57 <= -1
 vx50 + vy50 - vx58 - vy58 <= -1
 vx51 + vy51 - vx57 - vy57 <= -1
 vx51 + vy51 - vx58 - vy58 <= -1
 vx50 + vy50 - vx58 - vy58 <= -1
 vx50 + vy50 - vx59 - vy59 <= -1
 vx51 + vy51 - vx58 - vy58 <= -1
 vx51 + vy51 - vx59 - vy59 <= -1
 vx50 - vy50 - vx59 + vy59 >= 1
 vx50 - vy50 - vx60 + vy60 >= 1
 vx51 - vy51 - vx59 + vy59 >= 1
 vx51 - vy51 - vx60 + vy60 >= 1
 vx50 - vy50 - vx60 + vy60 >= 1
 vx50 - vy50 - vx61 + vy61 >= 1
 vx51 - vy51 - vx60 + vy60 >= 1
 vx51 - vy51 - vx61 + vy61 >= 1
 q50 - oa50 - ob50 - oc50 - od50 = 0
 a52 + a53 - 2 ua50 - oa50 = 0
 b52 + b53 - 2 ub50 - ob50 = 0
 c52 + c53 - 2 uc50 - oc50 = 0
 d52 + d53 - 2 ud50 - od50 = 0
 vx51 + vy51 - vx62 - vy62 <= -1
 vx51 + vy51 - vx53 - vy53 <= -1
 vx52 + vy52 - vx62 - vy62 <= -1
 vx52 + vy52 - vx53 - vy53 <= -1
 vx51 + vy51 - vx53 - vy53 <= -1
 vx51 + vy51 - vx54 - vy54 <= -1
 vx52 + vy52 - vx53 - vy53 <= -1
 vx52 + vy52 - vx54 - vy54 <= -1
 vx51 + vy51 - vx54 - vy54 <= -1
 vx51 + vy51 - vx55 - vy55 <= -1
 vx52 + vy52 - vx54 - vy54 <= -1
 vx52 + vy52 - vx55 - vy55 <= -1
 vx51 + vy51 - vx55 - vy55 <= -1
 vx51 + vy51 - vx56 - vy56 <= -1
 vx52 + vy52 - vx55 - vy55 <= -1
 vx52 + vy52 - vx56 - vy56 <= -1
 vx51 + vy51 - vx56 - vy56 <= -1
 vx51 + vy51 - vx57 - vy57 <= -1
 vx52 + vy52 - vx56 - vy56 <= -1
 vx52 + vy52 - vx57 - vy57 <= -1
 vx51 + vy51 - vx57 - vy57 <= -1
 vx51 + vy51 - vx58 - vy58 <= -1
 vx52 + vy52 - vx57 - vy57 <= -1
 vx52 + vy52 - vx58 - vy58 <= -1
 vx51 - vy51 - vx58 + vy58 >= 1
 vx51 - vy51 - vx59 + vy59 >= 1
 vx52 - vy52 - vx58 + vy58 >= 1
 vx52 - vy52 - vx59 + vy59 >= 1
 vx51 - vy51 - vx59 + vy59 >= 1
 vx51 - vy51 - vx60 + vy60 >= 1
 vx52 - vy52 - vx59 + vy59 >= 1
 vx52 - vy52 - vx60 + vy60 >= 1
 vx51 - vy51 - vx60 + vy60 >= 1
 vx51 - vy51 - vx61 + vy61 >= 1
 vx52 - vy52 - vx60 + vy60 >= 1
 vx52 - vy52 - vx61 + vy61 >= 1
 q51 - oa51 - ob51 - oc51 - od51 = 0
 a53 + a54 - 2 ua51 - oa51 = 0
 b53 + b54 - 2 ub51 - ob51 = 0
 c53 + c54 - 2 uc51 - oc51 = 0
 d53 + d54 - 2 ud51 - od51 = 0
 vx52 + vy52 - vx53 - vy53 <= -1
 vx52 + vy52 - vx54 - vy54 <= -1
 vx62 + vy62 - vx53 - vy53 <= -1
 vx62 + vy62 - vx54 - vy54 <= -1
 vx52 + vy52 - vx54 - vy54 <= -1
 vx52 + vy52 - vx55 - vy55 <= -1
 vx62 + vy62 - vx54 - vy54 <= -1
 vx62 + vy62 - vx55 - vy55 <= -1
 vx52 + vy52 - vx55 - vy55 <= -1
 vx52 + vy52 - vx56 - vy56 <= -1
 vx62 + vy62 - vx55 - vy55 <= -1
 vx62 + vy62 - vx56 - vy56 <= -1
 vx52 + vy52 - vx56 - vy56 <= -1
 vx52 + vy52 - vx57 - vy57 <= -1
 vx62 + vy62 - vx56 - vy56 <= -1
 vx62 + vy62 - vx57 - vy57 <= -1
 vx52 + vy52 - vx57 - vy57 <= -1
 vx52 + vy52 - vx58 - vy58 <= -1
 vx62 + vy62 - vx57 - vy57 <= -1
 vx62 + vy62 - vx58 - vy58 <= -1
 vx52 - vy52 - vx58 + vy58 >= 1
 vx52 - vy52 - vx59 + vy59 >= 1
 vx62 - vy62 - vx58 + vy58 >= 1
 vx62 - vy62 - vx59 + vy59 >= 1
 vx52 - vy52 - vx59 + vy59 >= 1
 vx52 - vy52 - vx60 + vy60 >= 1
 vx62 - vy62 - vx59 + vy59 >= 1
 vx62 - vy62 - vx60 + vy60 >= 1
 vx52 - vy52 - vx60 + vy60 >= 1
 vx52 - vy52 - vx61 + vy61 >= 1
 vx62 - vy62 - vx60 + vy60 >= 1
 vx62 - vy62 - vx61 + vy61 >= 1
 q52 - oa52 - ob52 - oc52 - od52 = 0
 a54 + a55 - 2 ua52 - oa52 = 0
 b54 + b55 - 2 ub52 - ob52 = 0
 c54 + c55 - 2 uc52 - oc52 = 0
 d54 + d55 - 2 ud52 - od52 = 0
 vx62 - vy62 - vx54 + vy54 >= 1
 vx62 - vy62 - vx55 + vy55 >= 1
 vx53 - vy53 - vx54 + vy54 >= 1
 vx53 - vy53 - vx55 + vy55 >= 1
 vx62 - vy62 - vx55 + vy55 >= 1
 vx62 - vy62 - vx56 + vy56 >= 1
 vx53 - vy53 - vx55 + vy55 >= 1
 vx53 - vy53 - vx56 + vy56 >= 1
 vx62 + vy62 - vx56 - vy56 <= -1
 vx62 + vy62 - vx57 - vy57 <= -1
 vx53 + vy53 - vx56 - vy56 <= -1
 vx53 + vy53 - vx57 - vy57 <= -1
 vx62 + vy62 - vx57 - vy57 <= -1
 vx62 + vy62 - vx58 - vy58 <= -1
 vx53 + vy53 - vx57 - vy57 <= -1
 vx53 + vy53 - vx58 - vy58 <= -1
 vx62 - vy62 - vx58 + vy58 >= 1
 vx62 - vy62 - vx59 + vy59 >= 1
 vx53 - vy53 - vx58 + vy58 >= 1
 vx53 - vy53 - vx59 + vy59 >= 1
 vx62 - vy62 - vx59 + vy59 >= 1
 vx62 - vy62 - vx60 + vy60 >= 1
 vx53 - vy53 - vx59 + vy59 >= 1
 vx53 - vy53 - vx60 + vy60 >= 1
 vx62 - vy62 - vx60 + vy60 >= 1
 vx62 - vy62 - vx61 + vy61 >= 1
 vx53 - vy53 - vx60 + vy60 >= 1
 vx53 - vy53 - vx61 + vy61 >= 1
 q53 - oa53 - ob53 - oc53 - od53 = 0
 a55 + a56 - 2 ua53 - oa53 = 0
 b55 + b56 - 2 ub53 - ob53 = 0
 c55 + c56 - 2 uc53 - oc53 = 0
 d55 + d56 - 2 ud53 - od53 = 0
 vx53 - vy53 - vx55 + vy55 >= 1
 vx53 - vy53 - vx56 + vy56 >= 1
 vx54 - vy54 - vx55 + vy55 >= 1
 vx54 - vy54 - vx56 + vy56 >= 1
 vx53 + vy53 - vx56 - vy56 <= -1
 vx53 + vy53 - vx57 - vy57 <= -1
 vx54 + vy54 - vx56 - vy56 <= -1
 vx54 + vy54 - vx57 - vy57 <= -1
 vx53 + vy53 - vx57 - vy57 <= -1
 vx53 + vy53 - vx58 - vy58 <= -1
 vx54 + vy54 - vx57 - vy57 <= -1
 vx54 + vy54 - vx58 - vy58 <= -1
 vx53 - vy53 - vx58 + vy58 >= 1
 vx53 - vy53 - vx59 + vy59 >= 1
 vx54 - vy54 - vx58 + vy58 >= 1
 vx54 - vy54 - vx59 + vy59 >= 1
 vx53 - vy53 - vx59 + vy59 >= 1
 vx53 - vy53 - vx60 + vy60 >= 1
 vx54 - vy54 - vx59 + vy59 >= 1
 vx54 - vy54 - vx60 + vy60 >= 1
 vx53 - vy53 - vx60 + vy60 >= 1
 vx53 - vy53 - vx61 + vy61 >= 1
 vx54 - vy54 - vx60 + vy60 >= 1
 vx54 - vy54 - vx61 + vy61 >= 1
 q54 - oa54 - ob54 - oc54 - od54 = 0
 a56 + a57 - 2 ua54 - oa54 = 0
 b56 + b57 - 2 ub54 - ob54 = 0
 c56 + c57 - 2 uc54 - oc54 = 0
 d56 + d57 - 2 ud54 - od54 = 0
 vx54 + vy54 - vx56 - vy56 <= -1
 vx54 + vy54 - vx57 - vy57 <= -1
 vx55 + vy55 - vx56 - vy56 <= -1
 vx55 + vy55 - vx57 - vy57 <= -1
 vx54 + vy54 - vx57 - vy57 <= -1
 vx54 + vy54 - vx58 - vy58 <= -1
 vx55 + vy55 - vx57 - vy57 <= -1
 vx55 + vy55 - vx58 - vy58 <= -1
 vx54 - vy54 - vx58 + vy58 >= 1
 vx54 - vy54 - vx59 + vy59 >= 1
 vx55 - vy55 - vx58 + vy58 >= 1
 vx55 - vy55 - vx59 + vy59 >= 1
 vx54 - vy54 - vx59 + vy59 >= 1
 vx54 - vy54 - vx60 + vy60 >= 1
 vx55 - vy55 - vx59 + vy59 >= 1
 vx55 - vy55 - vx60 + vy60 >= 1
 vx54 - vy54 - vx60 + vy60 >= 1
 vx54 - vy54 - vx61 + vy61 >= 1
 vx55 - vy55 - vx60 + vy60 >= 1
 vx55 - vy55 - vx61 + vy61 >= 1
 q55 - oa55 - ob55 - oc55 - od55 = 0
 a57 + a58 - 2 ua55 - oa55 = 0
 b57 + b58 - 2 ub55 - ob55 = 0
 c57 + c58 - 2 uc55 - oc55 = 0
 d57 + d58 - 2 ud55 - od55 = 0
 vx55 - vy55 - vx57 + vy57 >= 1
 vx55 - vy55 - vx58 + vy58 >= 1
 vx56 - vy56 - vx57 + vy57 >= 1
 vx56 - vy56 - vx58 + vy58 >= 1
 vx55 - vy55 - vx58 + vy58 >= 1
 vx55 - vy55 - vx59 + vy59 >= 1
 vx56 - vy56 - vx58 + vy58 >= 1
 vx56 - vy56 - vx59 + vy59 >= 1
 vx55 - vy55 - vx59 + vy59 >= 1
 vx55 - vy55 - vx60 + vy60 >= 1
 vx56 - vy56 - vx59 + vy59 >= 1
 vx56 - vy56 - vx60 + vy60 >= 1
 vx55 - vy55 - vx60 + vy60 >= 1
 vx55 - vy55 - vx61 + vy61 >= 1
 vx56 - vy56 - vx60 + vy60 >= 1
 vx56 - vy56 - vx61 + vy61 >= 1
 q56 - oa56 - ob56 - oc56 - od56 = 0
 a58 + a59 - 2 ua56 - oa56 = 0
 b58 + b59 - 2 ub56 - ob56 = 0
 c58 + c59 - 2 uc56 - oc56 = 0
 d58 + d59 - 2 ud56 - od56 = 0
 vx56 - vy56 - vx58 + vy58 >= 1
 vx56 - vy56 - vx59 + vy59 >= 1
 vx57 - vy57 - vx58 + vy58 >= 1
 vx57 - vy57 - vx59 + vy59 >= 1
 vx56 - vy56 - vx59 + vy59 >= 1
 vx56 - vy56 - vx60 + vy60 >= 1
 vx57 - vy57 - vx59 + vy59 >= 1
 vx57 - vy57 - vx60 + vy60 >= 1
 vx56 - vy56 - vx60 + vy60 >= 1
 vx56 - vy56 - vx61 + vy61 >= 1
 vx57 - vy57 - vx60 + vy60 >= 1
 vx57 - vy57 - vx61 + vy61 >= 1
 q57 - oa57 - ob57 - oc57 - od57 = 0
 a59 + a60 - 2 ua57 - oa57 = 0
 b59 + b60 - 2 ub57 - ob57 = 0
 c59 + c60 - 2 uc57 - oc57 = 0
 d59 + d60 - 2 ud57 - od57 = 0
 vx57 - vy57 - vx59 + vy59 >= 1
 vx57 - vy57 - vx60 + vy60 >= 1
 vx58 - vy58 - vx59 + vy59 >= 1
 vx58 - vy58 - vx60 + vy60 >= 1
 vx57 - vy57 - vx60 + vy60 >= 1
 vx57 - vy57 - vx61 + vy61 >= 1
 vx58 - vy58 - vx60 + vy60 >= 1
 vx58 - vy58 - vx61 + vy61 >= 1
 q58 - oa58 - ob58 - oc58 - od58 = 0
 a60 + a61 - 2 ua58 - oa58 = 0
 b60 + b61 - 2 ub58 - ob58 = 0
 c60 + c61 - 2 uc58 - oc58 = 0
 d60 + d61 - 2 ud58 - od58 = 0
 vx58 - vy58 - vx60 + vy60 >= 1
 vx58 - vy58 - vx61 + vy61 >= 1
 vx59 - vy59 - vx60 + vy60 >= 1
 vx59 - vy59 - vx61 + vy61 >= 1
 q59 - oa59 - ob59 - oc59 - od59 = 0
 a61 + a62 - 2 ua59 - oa59 = 0
 b61 + b62 - 2 ub59 - ob59 = 0
 c61 + c62 - 2 uc59 - oc59 = 0
 d61 + d62 - 2 ud59 - od59 = 0
Lazy Constraints
 3 a0 - 3 b0 + c0 - d0 + 3 a1 - 3 b1 + c1 - d1 - 9 h0 <= -0.5
 3 a0 - 3 b0 + c0 - d0 + 3 a1 - 3 b1 + c1 - d1 - 9 h0 >= -8.5
 3 a1 - 3 b1 + c1 - d1 + 3 a2 - 3 b2 + c2 - d2 - 9 h1 <= -0.5
 3 a1 - 3 b1 + c1 - d1 + 3 a2 - 3 b2 + c2 - d2 - 9 h1 >= -8.5
 3 a2 - 3 b2 + c2 - d2 + 3 a3 - 3 b3 + c3 - d3 - 9 h2 <= -0.5
 3 a2 - 3 b2 + c2 - d2 + 3 a3 - 3 b3 + c3 - d3 - 9 h2 >= -8.5
 3 a3 - 3 b3 + c3 - d3 + 3 a4 - 3 b4 + c4 - d4 - 9 h3 <= -0.5
 3 a3 - 3 b3 + c3 - d3 + 3 a4 - 3 b4 + c4 - d4 - 9 h3 >= -8.5
 3 a4 - 3 b4 + c4 - d4 + 3 a5 - 3 b5 + c5 - d5 - 9 h4 <= -0.5
 3 a4 - 3 b4 + c4 - d4 + 3 a5 - 3 b5 + c5 - d5 - 9 h4 >= -8.5
 3 a5 - 3 b5 + c5 - d5 + 3 a6 - 3 b6 + c6 - d6 - 9 h5 <= -0.5
 3 a5 - 3 b5 + c5 - d5 + 3 a6 - 3 b6 + c6 - d6 - 9 h5 >= -8.5
 3 a6 - 3 b6 + c6 - d6 + 3 a7 - 3 b7 + c7 - d7 - 9 h6 <= -0.5
 3 a6 - 3 b6 + c6 - d6 + 3 a7 - 3 b7 + c7 - d7 - 9 h6 >= -8.5
 3 a7 - 3 b7 + c7 - d7 + 3 a8 - 3 b8 + c8 - d8 - 9 h7 <= -0.5
 3 a7 - 3 b7 + c7 - d7 + 3 a8 - 3 b8 + c8 - d8 - 9 h7 >= -8.5
 3 a8 - 3 b8 + c8 - d8 + 3 a9 - 3 b9 + c9 - d9 - 9 h8 <= -0.5
 3 a8 - 3 b8 + c8 - d8 + 3 a9 - 3 b9 + c9 - d9 - 9 h8 >= -8.5
 3 a9 - 3 b9 + c9 - d9 + 3 a10 - 3 b10 + c10 - d10 - 9 h9 <= -0.5
 3 a9 - 3 b9 + c9 - d9 + 3 a10 - 3 b10 + c10 - d10 - 9 h9 >= -8.5
 3 a10 - 3 b10 + c10 - d10 + 3 a11 - 3 b11 + c11 - d11 - 9 h10 <= -0.5
 3 a10 - 3 b10 + c10 - d10 + 3 a11 - 3 b11 + c11 - d11 - 9 h10 >= -8.5
 3 a11 - 3 b11 + c11 - d11 + 3 a12 - 3 b12 + c12 - d12 - 9 h11 <= -0.5
 3 a11 - 3 b11 + c11 - d11 + 3 a12 - 3 b12 + c12 - d12 - 9 h11 >= -8.5
 3 a12 - 3 b12 + c12 - d12 + 3 a13 - 3 b13 + c13 - d13 - 9 h12 <= -0.5
 3 a12 - 3 b12 + c12 - d12 + 3 a13 - 3 b13 + c13 - d13 - 9 h12 >= -8.5
 3 a13 - 3 b13 + c13 - d13 + 3 a14 - 3 b14 + c14 - d14 - 9 h13 <= -0.5
 3 a13 - 3 b13 + c13 - d13 + 3 a14 - 3 b14 + c14 - d14 - 9 h13 >= -8.5
 3 a14 - 3 b14 + c14 - d14 + 3 a15 - 3 b15 + c15 - d15 - 9 h14 <= -0.5
 3 a14 - 3 b14 + c14 - d14 + 3 a15 - 3 b15 + c15 - d15 - 9 h14 >= -8.5
 3 a15 - 3 b15 + c15 - d15 + 3 a16 - 3 b16 + c16 - d16 - 9 h15 <= -0.5
 3 a15 - 3 b15 + c15 - d15 + 3 a16 - 3 b16 + c16 - d16 - 9 h15 >= -8.5
 3 a16 - 3 b16 + c16 - d16 + 3 a17 - 3 b17 + c17 - d17 - 9 h16 <= -0.5
 3 a16 - 3 b16 + c16 - d16 + 3 a17 - 3 b17 + c17 - d17 - 9 h16 >= -8.5
 3 a17 - 3 b17 + c17 - d17 + 3 a18 - 3 b18 + c18 - d18 - 9 h17 <= -0.5
 3 a17 - 3 b17 + c17 - d17 + 3 a18 - 3 b18 + c18 - d18 - 9 h17 >= -8.5
 3 a18 - 3 b18 + c18 - d18 + 3 a19 - 3 b19 + c19 - d19 - 9 h18 <= -0.5
 3 a18 - 3 b18 + c18 - d18 + 3 a19 - 3 b19 + c19 - d19 - 9 h18 >= -8.5
 3 a19 - 3 b19 + c19 - d19 + 3 a20 - 3 b20 + c20 - d20 - 9 h19 <= -0.5
 3 a19 - 3 b19 + c19 - d19 + 3 a20 - 3 b20 + c20 - d20 - 9 h19 >= -8.5
 3 a20 - 3 b20 + c20 - d20 + 3 a21 - 3 b21 + c21 - d21 - 9 h20 <= -0.5
 3 a20 - 3 b20 + c20 - d20 + 3 a21 - 3 b21 + c21 - d21 - 9 h20 >= -8.5
 3 a22 - 3 b22 + c22 - d22 + 3 a23 - 3 b23 + c23 - d23 - 9 h21 <= -0.5
 3 a22 - 3 b22 + c22 - d22 + 3 a23 - 3 b23 + c23 - d23 - 9 h21 >= -8.5
 3 a23 - 3 b23 + c23 - d23 + 3 a24 - 3 b24 + c24 - d24 - 9 h22 <= -0.5
 3 a23 - 3 b23 + c23 - d23 + 3 a24 - 3 b24 + c24 - d24 - 9 h22 >= -8.5
 3 a24 - 3 b24 + c24 - d24 + 3 a25 - 3 b25 + c25 - d25 - 9 h23 <= -0.5
 3 a24 - 3 b24 + c24 - d24 + 3 a25 - 3 b25 + c25 - d25 - 9 h23 >= -8.5
 3 a25 - 3 b25 + c25 - d25 + 3 a26 - 3 b26 + c26 - d26 - 9 h24 <= -0.5
 3 a25 - 3 b25 + c25 - d25 + 3 a26 - 3 b26 + c26 - d26 - 9 h24 >= -8.5
 3 a26 - 3 b26 + c26 - d26 + 3 a27 - 3 b27 + c27 - d27 - 9 h25 <= -0.5
 3 a26 - 3 b26 + c26 - d26 + 3 a27 - 3 b27 + c27 - d27 - 9 h25 >= -8.5
 3 a27 - 3 b27 + c27 - d27 + 3 a28 - 3 b28 + c28 - d28 - 9 h26 <= -0.5
 3 a27 - 3 b27 + c27 - d27 + 3 a28 - 3 b28 + c28 - d28 - 9 h26 >= -8.5
 3 a28 - 3 b28 + c28 - d28 + 3 a29 - 3 b29 + c29 - d29 - 9 h27 <= -0.5
 3 a28 - 3 b28 + c28 - d28 + 3 a29 - 3 b29 + c29 - d29 - 9 h27 >= -8.5
 3 a29 - 3 b29 + c29 - d29 + 3 a30 - 3 b30 + c30 - d30 - 9 h28 <= -0.5
 3 a29 - 3 b29 + c29 - d29 + 3 a30 - 3 b30 + c30 - d30 - 9 h28 >= -8.5
 3 a30 - 3 b30 + c30 - d30 + 3 a31 - 3 b31 + c31 - d31 - 9 h29 <= -0.5
 3 a30 - 3 b30 + c30 - d30 + 3 a31 - 3 b31 + c31 - d31 - 9 h29 >= -8.5
 3 a31 - 3 b31 + c31 - d31 + 3 a32 - 3 b32 + c32 - d32 - 9 h30 <= -0.5
 3 a31 - 3 b31 + c31 - d31 + 3 a32 - 3 b32 + c32 - d32 - 9 h30 >= -8.5
 3 a32 - 3 b32 + c32 - d32 + 3 a33 - 3 b33 + c33 - d33 - 9 h31 <= -0.5
 3 a32 - 3 b32 + c32 - d32 + 3 a33 - 3 b33 + c33 - d33 - 9 h31 >= -8.5
 3 a33 - 3 b33 + c33 - d33 + 3 a34 - 3 b34 + c34 - d34 - 9 h32 <= -0.5
 3 a33 - 3 b33 + c33 - d33 + 3 a34 - 3 b34 + c34 - d34 - 9 h32 >= -8.5
 3 a34 - 3 b34 + c34 - d34 + 3 a35 - 3 b35 + c35 - d35 - 9 h33 <= -0.5
 3 a34 - 3 b34 + c34 - d34 + 3 a35 - 3 b35 + c35 - d35 - 9 h33 >= -8.5
 3 a35 - 3 b35 + c35 - d35 + 3 a36 - 3 b36 + c36 - d36 - 9 h34 <= -0.5
 3 a35 - 3 b35 + c35 - d35 + 3 a36 - 3 b36 + c36 - d36 - 9 h34 >= -8.5
 3 a36 - 3 b36 + c36 - d36 + 3 a37 - 3 b37 + c37 - d37 - 9 h35 <= -0.5
 3 a36 - 3 b36 + c36 - d36 + 3 a37 - 3 b37 + c37 - d37 - 9 h35 >= -8.5
 3 a37 - 3 b37 + c37 - d37 + 3 a38 - 3 b38 + c38 - d38 - 9 h36 <= -0.5
 3 a37 - 3 b37 + c37 - d37 + 3 a38 - 3 b38 + c38 - d38 - 9 h36 >= -8.5
 3 a38 - 3 b38 + c38 - d38 + 3 a39 - 3 b39 + c39 - d39 - 9 h37 <= -0.5
 3 a38 - 3 b38 + c38 - d38 + 3 a39 - 3 b39 + c39 - d39 - 9 h37 >= -8.5
 3 a40 - 3 b40 + c40 - d40 + 3 a41 - 3 b41 + c41 - d41 - 9 h38 <= -0.5
 3 a40 - 3 b40 + c40 - d40 + 3 a41 - 3 b41 + c41 - d41 - 9 h38 >= -8.5
 3 a41 - 3 b41 + c41 - d41 + 3 a42 - 3 b42 + c42 - d42 - 9 h39 <= -0.5
 3 a41 - 3 b41 + c41 - d41 + 3 a42 - 3 b42 + c42 - d42 - 9 h39 >= -8.5
 3 a42 - 3 b42 + c42 - d42 + 3 a43 - 3 b43 + c43 - d43 - 9 h40 <= -0.5
 3 a42 - 3 b42 + c42 - d42 + 3 a43 - 3 b43 + c43 - d43 - 9 h40 >= -8.5
 3 a43 - 3 b43 + c43 - d43 + 3 a44 - 3 b44 + c44 - d44 - 9 h41 <= -0.5
 3 a43 - 3 b43 + c43 - d43 + 3 a44 - 3 b44 + c44 - d44 - 9 h41 >= -8.5
 3 a44 - 3 b44 + c44 - d44 + 3 a45 - 3 b45 + c45 - d45 - 9 h42 <= -0.5
 3 a44 - 3 b44 + c44 - d44 + 3 a45 - 3 b45 + c45 - d45 - 9 h42 >= -8.5
 3 a45 - 3 b45 + c45 - d45 + 3 a46 - 3 b46 + c46 - d46 - 9 h43 <= -0.5
 3 a45 - 3 b45 + c45 - d45 + 3 a46 - 3 b46 + c46 - d46 - 9 h43 >= -8.5
 3 a46 - 3 b46 + c46 - d46 + 3 a47 - 3 b47 + c47 - d47 - 9 h44 <= -0.5
 3 a46 - 3 b46 + c46 - d46 + 3 a47 - 3 b47 + c47 - d47 - 9 h44 >= -8.5
 3 a47 - 3 b47 + c47 - d47 + 3 a48 - 3 b48 + c48 - d48 - 9 h45 <= -0.5
 3 a47 - 3 b47 + c47 - d47 + 3 a48 - 3 b48 + c48 - d48 - 9 h45 >= -8.5
 3 a48 - 3 b48 + c48 - d48 + 3 a49 - 3 b49 + c49 - d49 - 9 h46 <= -0.5
 3 a48 - 3 b48 + c48 - d48 + 3 a49 - 3 b49 + c49 - d49 - 9 h46 >= -8.5
 3 a49 - 3 b49 + c49 - d49 + 3 a50 - 3 b50 + c50 - d50 - 9 h47 <= -0.5
 3 a49 - 3 b49 + c49 - d49 + 3 a50 - 3 b50 + c50 - d50 - 9 h47 >= -8.5
 3 a50 - 3 b50 + c50 - d50 + 3 a51 - 3 b51 + c51 - d51 - 9 h48 <= -0.5
 3 a50 - 3 b50 + c50 - d50 + 3 a51 - 3 b51 + c51 - d51 - 9 h48 >= -8.5
 3 a51 - 3 b51 + c51 - d51 + 3 a52 - 3 b52 + c52 - d52 - 9 h49 <= -0.5
 3 a51 - 3 b51 + c51 - d51 + 3 a52 - 3 b52 + c52 - d52 - 9 h49 >= -8.5
 3 a52 - 3 b52 + c52 - d52 + 3 a53 - 3 b53 + c53 - d53 - 9 h50 <= -0.5
 3 a52 - 3 b52 + c52 - d52 + 3 a53 - 3 b53 + c53 - d53 - 9 h50 >= -8.5
 3 a53 - 3 b53 + c53 - d53 + 3 a54 - 3 b54 + c54 - d54 - 9 h51 <= -0.5
 3 a53 - 3 b53 + c53 - d53 + 3 a54 - 3 b54 + c54 - d54 - 9 h51 >= -8.5
 3 a54 - 3 b54 + c54 - d54 + 3 a55 - 3 b55 + c55 - d55 - 9 h52 <= -0.5
 3 a54 - 3 b54 + c54 - d54 + 3 a55 - 3 b55 + c55 - d55 - 9 h52 >= -8.5
 3 a55 - 3 b55 + c55 - d55 + 3 a56 - 3 b56 + c56 - d56 - 9 h53 <= -0.5
 3 a55 - 3 b55 + c55 - d55 + 3 a56 - 3 b56 + c56 - d56 - 9 h53 >= -8.5
 3 a56 - 3 b56 + c56 - d56 + 3 a57 - 3 b57 + c57 - d57 - 9 h54 <= -0.5
 3 a56 - 3 b56 + c56 - d56 + 3 a57 - 3 b57 + c57 - d57 - 9 h54 >= -8.5
 3 a57 - 3 b57 + c57 - d57 + 3 a58 - 3 b58 + c58 - d58 - 9 h55 <= -0.5
 3 a57 - 3 b57 + c57 - d57 + 3 a58 - 3 b58 + c58 - d58 - 9 h55 >= -8.5
 3 a58 - 3 b58 + c58 - d58 + 3 a59 - 3 b59 + c59 - d59 - 9 h56 <= -0.5
 3 a58 - 3 b58 + c58 - d58 + 3 a59 - 3 b59 + c59 - d59 - 9 h56 >= -8.5
 3 a59 - 3 b59 + c59 - d59 + 3 a60 - 3 b60 + c60 - d60 - 9 h57 <= -0.5
 3 a59 - 3 b59 + c59 - d59 + 3 a60 - 3 b60 + c60 - d60 - 9 h57 >= -8.5
 3 a60 - 3 b60 + c60 - d60 + 3 a61 - 3 b61 + c61 - d61 - 9 h58 <= -0.5
 3 a60 - 3 b60 + c60 - d60 + 3 a61 - 3 b61 + c61 - d61 - 9 h58 >= -8.5
 3 a61 - 3 b61 + c61 - d61 + 3 a62 - 3 b62 + c62 - d62 - 9 h59 <= -0.5
 3 a61 - 3 b61 + c61 - d61 + 3 a62 - 3 b62 + c62 - d62 - 9 h59 >= -8.5
Bounds
 1 <= l0 <= 8
 1 <= l1 <= 8
 1 <= l2 <= 8
 1 <= l3 <= 8
 1 <= l4 <= 8
 1 <= l5 <= 8
 1 <= l6 <= 8
 1 <= l7 <= 8
 1 <= l8 <= 8
 1 <= l9 <= 8
 1 <= l10 <= 8
 1 <= l11 <= 8
 1 <= l12 <= 8
 1 <= l13 <= 8
 1 <= l14 <= 8
 1 <= l15 <= 8
 1 <= l16 <= 8
 1 <= l17 <= 8
 1 <= l18 <= 8
 1 <= l19 <= 8
 1 <= l20 <= 8
 1 <= l21 <= 8
 1 <= l22 <= 8
 1 <= l23 <= 8
 1 <= l24 <= 8
 1 <= l25 <= 8
 1 <= l26 <= 8
 1 <= l27 <= 8
 1 <= l28 <= 8
 1 <= l29 <= 8
 1 <= l30 <= 8
 1 <= l31 <= 8
 1 <= l32 <= 8
 1 <= l33 <= 8
 1 <= l34 <= 8
 1 <= l35 <= 8
 1 <= l36 <= 8
 1 <= l37 <= 8
 1 <= l38 <= 8
 1 <= l39 <= 8
 1 <= l40 <= 8
 1 <= l41 <= 8
 1 <= l42 <= 8
 1 <= l43 <= 8
 1 <= l44 <= 8
 1 <= l45 <= 8
 1 <= l46 <= 8
 1 <= l47 <= 8
 1 <= l48 <= 8
 1 <= l49 <= 8
 1 <= l50 <= 8
 1 <= l51 <= 8
 1 <= l52 <= 8
 1 <= l53 <= 8
 1 <= l54 <= 8
 1 <= l55 <= 8
 1 <= l56 <= 8
 1 <= l57 <= 8
 1 <= l58 <= 8
 1 <= l59 <= 8
 1 <= l60 <= 8
 1 <= l61 <= 8
 1 <= l62 <= 8
 9850 <= vx0 <= 10150
 9850 <= vx1 <= 10150
 9850 <= vx2 <= 10150
 9850 <= vx3 <= 10150
 9850 <= vx4 <= 10150
 9850 <= vx5 <= 10150
 9850 <= vx6 <= 10150
 9850 <= vx7 <= 10150
 9850 <= vx8 <= 10150
 9850 <= vx9 <= 10150
 9850 <= vx10 <= 10150
 9850 <= vx11 <= 10150
 9850 <= vx12 <= 10150
 9850 <= vx13 <= 10150
 9850 <= vx14 <= 10150
 9850 <= vx15 <= 10150
 9850 <= vx16 <= 10150
 9850 <= vx17 <= 10150
 9850 <= vx18 <= 10150
 9850 <= vx19 <= 10150
 9850 <= vx20 <= 10150
 9850 <= vx21 <= 10150
 9850 <= vx22 <= 10150
 9850 <= vx23 <= 10150
 9850 <= vx24 <= 10150
 9850 <= vx25 <= 10150
 9850 <= vx26 <= 10150
 9850 <= vx27 <= 10150
 9850 <= vx28 <= 10150
 9850 <= vx29 <= 10150
 9850 <= vx30 <= 10150
 9850 <= vx31 <= 10150
 9850 <= vx32 <= 10150
 9850 <= vx33 <= 10150
 9850 <= vx34 <= 10150
 9850 <= vx35 <= 10150
 9850 <= vx36 <= 10150
 9850 <= vx37 <= 10150
 9850 <= vx38 <= 10150
 9850 <= vx39 <= 10150
 9850 <= vx40 <= 10150
 9850 <= vx41 <= 10150
 9850 <= vx42 <= 10150
 9850 <= vx43 <= 10150
 9850 <= vx44 <= 10150
 9850 <= vx45 <= 10150
 9850 <= vx46 <= 10150
 9850 <= vx47 <= 10150
 9850 <= vx48 <= 10150
 9850 <= vx49 <= 10150
 9850 <= vx50 <= 10150
 9850 <= vx51 <= 10150
 9850 <= vx52 <= 10150
 9850 <= vx53 <= 10150
 9850 <= vx54 <= 10150
 9850 <= vx55 <= 10150
 9850 <= vx56 <= 10150
 9850 <= vx57 <= 10150
 9850 <= vx58 <= 10150
 9850 <= vx59 <= 10150
 9850 <= vx60 <= 10150
 9850 <= vx61 <= 10150
 9850 <= vx62 <= 10150
 9850 <= vx63 <= 10150
 9850 <= vx64 <= 10150
 9850 <= vx65 <= 10150
 9850 <= vx17 <= 10150
 9850 <= vx27 <= 10150
 9850 <= vx30 <= 10150
 9850 <= vx31 <= 10150
 9850 <= vx33 <= 10150
 9850 <= vx37 <= 10150
 9850 <= vx38 <= 10150
 9850 <= vx39 <= 10150
 9850 <= vx36 <= 10150
 9850 <= vx13 <= 10150
 9850 <= vx4 <= 10150
 9850 <= vx29 <= 10150
 9850 <= vx34 <= 10150
 9850 <= vx50 <= 10150
 9850 <= vx32 <= 10150
 9850 <= vx49 <= 10150
 9850 <= vx35 <= 10150
 9850 <= vx65 <= 10150
 9850 <= vx28 <= 10150
 9850 <= vx48 <= 10150
 9850 <= vx47 <= 10150
 9850 <= vx46 <= 10150
 9850 <= vx45 <= 10150
 9850 <= vx51 <= 10150
 9850 <= vx52 <= 10150
 9850 <= vx18 <= 10150
 9850 <= vx19 <= 10150
 9850 <= vx1 <= 10150
 9850 <= vx26 <= 10150
 9850 <= vx11 <= 10150
 9850 <= vx8 <= 10150
 9850 <= vx62 <= 10150
 9850 <= vx6 <= 10150
 9850 <= vx53 <= 10150
 9850 <= vx54 <= 10150
 9850 <= vx7 <= 10150
 9850 <= vx1 <= 10150
 9850 <= vx9 <= 10150
 9850 <= vx21 <= 10150
 9850 <= vx24 <= 10150
 9850 <= vx14 <= 10150
 9850 <= vx2 <= 10150
 9850 <= vx0 <= 10150
 9850 <= vx12 <= 10150
 9850 <= vx10 <= 10150
 9850 <= vx16 <= 10150
 9850 <= vx59 <= 10150
 9850 <= vx61 <= 10150
 9850 <= vx60 <= 10150
 9850 <= vx55 <= 10150
 9850 <= vx56 <= 10150
 9850 <= vx23 <= 10150
 9850 <= vx20 <= 10150
 9850 <= vx57 <= 10150
 9850 <= vx58 <= 10150
 9850 <= vx25 <= 10150
 9850 <= vx22 <= 10150
 9850 <= vx5 <= 10150
 9850 <= vx42 <= 10150
 9850 <= vx42 <= 10150
 9850 <= vx43 <= 10150
 9850 <= vx3 <= 10150
 9850 <= vx63 <= 10150
 9850 <= vx44 <= 10150
 9850 <= vx41 <= 10150
 9850 <= vx40 <= 10150
 9850 <= vx64 <= 10150
 9850 <= vx15 <= 10150
 9850 <= vy0 <= 10150
 9850 <= vy1 <= 10150
 9850 <= vy2 <= 10150
 9850 <= vy3 <= 10150
 9850 <= vy4 <= 10150
 9850 <= vy5 <= 10150
 9850 <= vy6 <= 10150
 9850 <= vy7 <= 10150
 9850 <= vy8 <= 10150
 9850 <= vy9 <= 10150
 9850 <= vy10 <= 10150
 9850 <= vy11 <= 10150
 9850 <= vy12 <= 10150
 9850 <= vy13 <= 10150
 9850 <= vy14 <= 10150
 9850 <= vy15 <= 10150
 9850 <= vy16 <= 10150
 9850 <= vy17 <= 10150
 9850 <= vy18 <= 10150
 9850 <= vy19 <= 10150
 9850 <= vy20 <= 10150
 9850 <= vy21 <= 10150
 9850 <= vy22 <= 10150
 9850 <= vy23 <= 10150
 9850 <= vy24 <= 10150
 9850 <= vy25 <= 10150
 9850 <= vy26 <= 10150
 9850 <= vy27 <= 10150
 9850 <= vy28 <= 10150
 9850 <= vy29 <= 10150
 9850 <= vy30 <= 10150
 9850 <= vy31 <= 10150
 9850 <= vy32 <= 10150
 9850 <= vy33 <= 10150
 9850 <= vy34 <= 10150
 9850 <= vy35 <= 10150
 9850 <= vy36 <= 10150
 9850 <= vy37 <= 10150
 9850 <= vy38 <= 10150
 9850 <= vy39 <= 10150
 9850 <= vy40 <= 10150
 9850 <= vy41 <= 10150
 9850 <= vy42 <= 10150
 9850 <= vy43 <= 10150
 9850 <= vy44 <= 10150
 9850 <= vy45 <= 10150
 9850 <= vy46 <= 10150
 9850 <= vy47 <= 10150
 9850 <= vy48 <= 10150
 9850 <= vy49 <= 10150
 9850 <= vy50 <= 10150
 9850 <= vy51 <= 10150
 9850 <= vy52 <= 10150
 9850 <= vy53 <= 10150
 9850 <= vy54 <= 10150
 9850 <= vy55 <= 10150
 9850 <= vy56 <= 10150
 9850 <= vy57 <= 10150
 9850 <= vy58 <= 10150
 9850 <= vy59 <= 10150
 9850 <= vy60 <= 10150
 9850 <= vy61 <= 10150
 9850 <= vy62 <= 10150
 9850 <= vy63 <= 10150
 9850 <= vy64 <= 10150
 9850 <= vy65 <= 10150
 9850 <= vy17 <= 10150
 9850 <= vy27 <= 10150
 9850 <= vy30 <= 10150
 9850 <= vy31 <= 10150
 9850 <= vy33 <= 10150
 9850 <= vy37 <= 10150
 9850 <= vy38 <= 10150
 9850 <= vy39 <= 10150
 9850 <= vy36 <= 10150
 9850 <= vy13 <= 10150
 9850 <= vy4 <= 10150
 9850 <= vy29 <= 10150
 9850 <= vy34 <= 10150
 9850 <= vy50 <= 10150
 9850 <= vy32 <= 10150
 9850 <= vy49 <= 10150
 9850 <= vy35 <= 10150
 9850 <= vy65 <= 10150
 9850 <= vy28 <= 10150
 9850 <= vy48 <= 10150
 9850 <= vy47 <= 10150
 9850 <= vy46 <= 10150
 9850 <= vy45 <= 10150
 9850 <= vy51 <= 10150
 9850 <= vy52 <= 10150
 9850 <= vy18 <= 10150
 9850 <= vy19 <= 10150
 9850 <= vy1 <= 10150
 9850 <= vy26 <= 10150
 9850 <= vy11 <= 10150
 9850 <= vy8 <= 10150
 9850 <= vy62 <= 10150
 9850 <= vy6 <= 10150
 9850 <= vy53 <= 10150
 9850 <= vy54 <= 10150
 9850 <= vy7 <= 10150
 9850 <= vy1 <= 10150
 9850 <= vy9 <= 10150
 9850 <= vy21 <= 10150
 9850 <= vy24 <= 10150
 9850 <= vy14 <= 10150
 9850 <= vy2 <= 10150
 9850 <= vy0 <= 10150
 9850 <= vy12 <= 10150
 9850 <= vy10 <= 10150
 9850 <= vy16 <= 10150
 9850 <= vy59 <= 10150
 9850 <= vy61 <= 10150
 9850 <= vy60 <= 10150
 9850 <= vy55 <= 10150
 9850 <= vy56 <= 10150
 9850 <= vy23 <= 10150
 9850 <= vy20 <= 10150
 9850 <= vy57 <= 10150
 9850 <= vy58 <= 10150
 9850 <= vy25 <= 10150
 9850 <= vy22 <= 10150
 9850 <= vy5 <= 10150
 9850 <= vy42 <= 10150
 9850 <= vy42 <= 10150
 9850 <= vy43 <= 10150
 9850 <= vy3 <= 10150
 9850 <= vy63 <= 10150
 9850 <= vy44 <= 10150
 9850 <= vy41 <= 10150
 9850 <= vy40 <= 10150
 9850 <= vy64 <= 10150
 9850 <= vy15 <= 10150
 0 <= pa0
 0 <= pa1
 0 <= pa2
 0 <= pa3
 0 <= pa4
 0 <= pa5
 0 <= pa6
 0 <= pa7
 0 <= pa8
 0 <= pa9
 0 <= pa10
 0 <= pa11
 0 <= pa12
 0 <= pa13
 0 <= pa14
 0 <= pa15
 0 <= pa16
 0 <= pa17
 0 <= pa18
 0 <= pa19
 0 <= pa20
 0 <= pa21
 0 <= pa22
 0 <= pa23
 0 <= pa24
 0 <= pa25
 0 <= pa26
 0 <= pa27
 0 <= pa28
 0 <= pa29
 0 <= pa30
 0 <= pa31
 0 <= pa32
 0 <= pa33
 0 <= pa34
 0 <= pa35
 0 <= pa36
 0 <= pa37
 0 <= pa38
 0 <= pa39
 0 <= pa40
 0 <= pa41
 0 <= pa42
 0 <= pa43
 0 <= pa44
 0 <= pa45
 0 <= pa46
 0 <= pa47
 0 <= pa48
 0 <= pa49
 0 <= pa50
 0 <= pa51
 0 <= pa52
 0 <= pa53
 0 <= pa54
 0 <= pa55
 0 <= pa56
 0 <= pa57
 0 <= pa58
 0 <= pa59
 0 <= pa60
 0 <= pa61
 0 <= pa62
 0 <= pb0
 0 <= pb1
 0 <= pb2
 0 <= pb3
 0 <= pb4
 0 <= pb5
 0 <= pb6
 0 <= pb7
 0 <= pb8
 0 <= pb9
 0 <= pb10
 0 <= pb11
 0 <= pb12
 0 <= pb13
 0 <= pb14
 0 <= pb15
 0 <= pb16
 0 <= pb17
 0 <= pb18
 0 <= pb19
 0 <= pb20
 0 <= pb21
 0 <= pb22
 0 <= pb23
 0 <= pb24
 0 <= pb25
 0 <= pb26
 0 <= pb27
 0 <= pb28
 0 <= pb29
 0 <= pb30
 0 <= pb31
 0 <= pb32
 0 <= pb33
 0 <= pb34
 0 <= pb35
 0 <= pb36
 0 <= pb37
 0 <= pb38
 0 <= pb39
 0 <= pb40
 0 <= pb41
 0 <= pb42
 0 <= pb43
 0 <= pb44
 0 <= pb45
 0 <= pb46
 0 <= pb47
 0 <= pb48
 0 <= pb49
 0 <= pb50
 0 <= pb51
 0 <= pb52
 0 <= pb53
 0 <= pb54
 0 <= pb55
 0 <= pb56
 0 <= pb57
 0 <= pb58
 0 <= pb59
 0 <= pb60
 0 <= pb61
 0 <= pb62
 0 <= pc0
 0 <= pc1
 0 <= pc2
 0 <= pc3
 0 <= pc4
 0 <= pc5
 0 <= pc6
 0 <= pc7
 0 <= pc8
 0 <= pc9
 0 <= pc10
 0 <= pc11
 0 <= pc12
 0 <= pc13
 0 <= pc14
 0 <= pc15
 0 <= pc16
 0 <= pc17
 0 <= pc18
 0 <= pc19
 0 <= pc20
 0 <= pc21
 0 <= pc22
 0 <= pc23
 0 <= pc24
 0 <= pc25
 0 <= pc26
 0 <= pc27
 0 <= pc28
 0 <= pc29
 0 <= pc30
 0 <= pc31
 0 <= pc32
 0 <= pc33
 0 <= pc34
 0 <= pc35
 0 <= pc36
 0 <= pc37
 0 <= pc38
 0 <= pc39
 0 <= pc40
 0 <= pc41
 0 <= pc42
 0 <= pc43
 0 <= pc44
 0 <= pc45
 0 <= pc46
 0 <= pc47
 0 <= pc48
 0 <= pc49
 0 <= pc50
 0 <= pc51
 0 <= pc52
 0 <= pc53
 0 <= pc54
 0 <= pc55
 0 <= pc56
 0 <= pc57
 0 <= pc58
 0 <= pc59
 0 <= pc60
 0 <= pc61
 0 <= pc62
 0 <= pd0
 0 <= pd1
 0 <= pd2
 0 <= pd3
 0 <= pd4
 0 <= pd5
 0 <= pd6
 0 <= pd7
 0 <= pd8
 0 <= pd9
 0 <= pd10
 0 <= pd11
 0 <= pd12
 0 <= pd13
 0 <= pd14
 0 <= pd15
 0 <= pd16
 0 <= pd17
 0 <= pd18
 0 <= pd19
 0 <= pd20
 0 <= pd21
 0 <= pd22
 0 <= pd23
 0 <= pd24
 0 <= pd25
 0 <= pd26
 0 <= pd27
 0 <= pd28
 0 <= pd29
 0 <= pd30
 0 <= pd31
 0 <= pd32
 0 <= pd33
 0 <= pd34
 0 <= pd35
 0 <= pd36
 0 <= pd37
 0 <= pd38
 0 <= pd39
 0 <= pd40
 0 <= pd41
 0 <= pd42
 0 <= pd43
 0 <= pd44
 0 <= pd45
 0 <= pd46
 0 <= pd47
 0 <= pd48
 0 <= pd49
 0 <= pd50
 0 <= pd51
 0 <= pd52
 0 <= pd53
 0 <= pd54
 0 <= pd55
 0 <= pd56
 0 <= pd57
 0 <= pd58
 0 <= pd59
 0 <= pd60
 0 <= pd61
 0 <= pd62
 0 <= q0 <= 3
 0 <= q1 <= 3
 0 <= q2 <= 3
 0 <= q3 <= 3
 0 <= q4 <= 3
 0 <= q5 <= 3
 0 <= q6 <= 3
 0 <= q7 <= 3
 0 <= q8 <= 3
 0 <= q9 <= 3
 0 <= q10 <= 3
 0 <= q11 <= 3
 0 <= q12 <= 3
 0 <= q13 <= 3
 0 <= q14 <= 3
 0 <= q15 <= 3
 0 <= q16 <= 3
 0 <= q17 <= 3
 0 <= q18 <= 3
 0 <= q19 <= 3
 0 <= q20 <= 3
 0 <= q21 <= 3
 0 <= q22 <= 3
 0 <= q23 <= 3
 0 <= q24 <= 3
 0 <= q25 <= 3
 0 <= q26 <= 3
 0 <= q27 <= 3
 0 <= q28 <= 3
 0 <= q29 <= 3
 0 <= q30 <= 3
 0 <= q31 <= 3
 0 <= q32 <= 3
 0 <= q33 <= 3
 0 <= q34 <= 3
 0 <= q35 <= 3
 0 <= q36 <= 3
 0 <= q37 <= 3
 0 <= q38 <= 3
 0 <= q39 <= 3
 0 <= q40 <= 3
 0 <= q41 <= 3
 0 <= q42 <= 3
 0 <= q43 <= 3
 0 <= q44 <= 3
 0 <= q45 <= 3
 0 <= q46 <= 3
 0 <= q47 <= 3
 0 <= q48 <= 3
 0 <= q49 <= 3
 0 <= q50 <= 3
 0 <= q51 <= 3
 0 <= q52 <= 3
 0 <= q53 <= 3
 0 <= q54 <= 3
 0 <= q55 <= 3
 0 <= q56 <= 3
 0 <= q57 <= 3
 0 <= q58 <= 3
 0 <= q59 <= 3
General
 q0
 q1
 q2
 q3
 q4
 q5
 q6
 q7
 q8
 q9
 q10
 q11
 q12
 q13
 q14
 q15
 q16
 q17
 q18
 q19
 q20
 q21
 q22
 q23
 q24
 q25
 q26
 q27
 q28
 q29
 q30
 q31
 q32
 q33
 q34
 q35
 q36
 q37
 q38
 q39
 q40
 q41
 q42
 q43
 q44
 q45
 q46
 q47
 q48
 q49
 q50
 q51
 q52
 q53
 q54
 q55
 q56
 q57
 q58
 q59
Binary
 a0
 a1
 a2
 a3
 a4
 a5
 a6
 a7
 a8
 a9
 a10
 a11
 a12
 a13
 a14
 a15
 a16
 a17
 a18
 a19
 a20
 a21
 a22
 a23
 a24
 a25
 a26
 a27
 a28
 a29
 a30
 a31
 a32
 a33
 a34
 a35
 a36
 a37
 a38
 a39
 a40
 a41
 a42
 a43
 a44
 a45
 a46
 a47
 a48
 a49
 a50
 a51
 a52
 a53
 a54
 a55
 a56
 a57
 a58
 a59
 a60
 a61
 a62
 b0
 b1
 b2
 b3
 b4
 b5
 b6
 b7
 b8
 b9
 b10
 b11
 b12
 b13
 b14
 b15
 b16
 b17
 b18
 b19
 b20
 b21
 b22
 b23
 b24
 b25
 b26
 b27
 b28
 b29
 b30
 b31
 b32
 b33
 b34
 b35
 b36
 b37
 b38
 b39
 b40
 b41
 b42
 b43
 b44
 b45
 b46
 b47
 b48
 b49
 b50
 b51
 b52
 b53
 b54
 b55
 b56
 b57
 b58
 b59
 b60
 b61
 b62
 c0
 c1
 c2
 c3
 c4
 c5
 c6
 c7
 c8
 c9
 c10
 c11
 c12
 c13
 c14
 c15
 c16
 c17
 c18
 c19
 c20
 c21
 c22
 c23
 c24
 c25
 c26
 c27
 c28
 c29
 c30
 c31
 c32
 c33
 c34
 c35
 c36
 c37
 c38
 c39
 c40
 c41
 c42
 c43
 c44
 c45
 c46
 c47
 c48
 c49
 c50
 c51
 c52
 c53
 c54
 c55
 c56
 c57
 c58
 c59
 c60
 c61
 c62
 d0
 d1
 d2
 d3
 d4
 d5
 d6
 d7
 d8
 d9
 d10
 d11
 d12
 d13
 d14
 d15
 d16
 d17
 d18
 d19
 d20
 d21
 d22
 d23
 d24
 d25
 d26
 d27
 d28
 d29
 d30
 d31
 d32
 d33
 d34
 d35
 d36
 d37
 d38
 d39
 d40
 d41
 d42
 d43
 d44
 d45
 d46
 d47
 d48
 d49
 d50
 d51
 d52
 d53
 d54
 d55
 d56
 d57
 d58
 d59
 d60
 d61
 d62
 h0
 h1
 h2
 h3
 h4
 h5
 h6
 h7
 h8
 h9
 h10
 h11
 h12
 h13
 h14
 h15
 h16
 h17
 h18
 h19
 h20
 h21
 h22
 h23
 h24
 h25
 h26
 h27
 h28
 h29
 h30
 h31
 h32
 h33
 h34
 h35
 h36
 h37
 h38
 h39
 h40
 h41
 h42
 h43
 h44
 h45
 h46
 h47
 h48
 h49
 h50
 h51
 h52
 h53
 h54
 h55
 h56
 h57
 h58
 h59
 oa0
 oa1
 oa2
 oa3
 oa4
 oa5
 oa6
 oa7
 oa8
 oa9
 oa10
 oa11
 oa12
 oa13
 oa14
 oa15
 oa16
 oa17
 oa18
 oa19
 oa20
 oa21
 oa22
 oa23
 oa24
 oa25
 oa26
 oa27
 oa28
 oa29
 oa30
 oa31
 oa32
 oa33
 oa34
 oa35
 oa36
 oa37
 oa38
 oa39
 oa40
 oa41
 oa42
 oa43
 oa44
 oa45
 oa46
 oa47
 oa48
 oa49
 oa50
 oa51
 oa52
 oa53
 oa54
 oa55
 oa56
 oa57
 oa58
 oa59
 ob0
 ob1
 ob2
 ob3
 ob4
 ob5
 ob6
 ob7
 ob8
 ob9
 ob10
 ob11
 ob12
 ob13
 ob14
 ob15
 ob16
 ob17
 ob18
 ob19
 ob20
 ob21
 ob22
 ob23
 ob24
 ob25
 ob26
 ob27
 ob28
 ob29
 ob30
 ob31
 ob32
 ob33
 ob34
 ob35
 ob36
 ob37
 ob38
 ob39
 ob40
 ob41
 ob42
 ob43
 ob44
 ob45
 ob46
 ob47
 ob48
 ob49
 ob50
 ob51
 ob52
 ob53
 ob54
 ob55
 ob56
 ob57
 ob58
 ob59
 oc0
 oc1
 oc2
 oc3
 oc4
 oc5
 oc6
 oc7
 oc8
 oc9
 oc10
 oc11
 oc12
 oc13
 oc14
 oc15
 oc16
 oc17
 oc18
 oc19
 oc20
 oc21
 oc22
 oc23
 oc24
 oc25
 oc26
 oc27
 oc28
 oc29
 oc30
 oc31
 oc32
 oc33
 oc34
 oc35
 oc36
 oc37
 oc38
 oc39
 oc40
 oc41
 oc42
 oc43
 oc44
 oc45
 oc46
 oc47
 oc48
 oc49
 oc50
 oc51
 oc52
 oc53
 oc54
 oc55
 oc56
 oc57
 oc58
 oc59
 od0
 od1
 od2
 od3
 od4
 od5
 od6
 od7
 od8
 od9
 od10
 od11
 od12
 od13
 od14
 od15
 od16
 od17
 od18
 od19
 od20
 od21
 od22
 od23
 od24
 od25
 od26
 od27
 od28
 od29
 od30
 od31
 od32
 od33
 od34
 od35
 od36
 od37
 od38
 od39
 od40
 od41
 od42
 od43
 od44
 od45
 od46
 od47
 od48
 od49
 od50
 od51
 od52
 od53
 od54
 od55
 od56
 od57
 od58
 od59
 ua0
 ua1
 ua2
 ua3
 ua4
 ua5
 ua6
 ua7
 ua8
 ua9
 ua10
 ua11
 ua12
 ua13
 ua14
 ua15
 ua16
 ua17
 ua18
 ua19
 ua20
 ua21
 ua22
 ua23
 ua24
 ua25
 ua26
 ua27
 ua28
 ua29
 ua30
 ua31
 ua32
 ua33
 ua34
 ua35
 ua36
 ua37
 ua38
 ua39
 ua40
 ua41
 ua42
 ua43
 ua44
 ua45
 ua46
 ua47
 ua48
 ua49
 ua50
 ua51
 ua52
 ua53
 ua54
 ua55
 ua56
 ua57
 ua58
 ua59
 ub0
 ub1
 ub2
 ub3
 ub4
 ub5
 ub6
 ub7
 ub8
 ub9
 ub10
 ub11
 ub12
 ub13
 ub14
 ub15
 ub16
 ub17
 ub18
 ub19
 ub20
 ub21
 ub22
 ub23
 ub24
 ub25
 ub26
 ub27
 ub28
 ub29
 ub30
 ub31
 ub32
 ub33
 ub34
 ub35
 ub36
 ub37
 ub38
 ub39
 ub40
 ub41
 ub42
 ub43
 ub44
 ub45
 ub46
 ub47
 ub48
 ub49
 ub50
 ub51
 ub52
 ub53
 ub54
 ub55
 ub56
 ub57
 ub58
 ub59
 uc0
 uc1
 uc2
 uc3
 uc4
 uc5
 uc6
 uc7
 uc8
 uc9
 uc10
 uc11
 uc12
 uc13
 uc14
 uc15
 uc16
 uc17
 uc18
 uc19
 uc20
 uc21
 uc22
 uc23
 uc24
 uc25
 uc26
 uc27
 uc28
 uc29
 uc30
 uc31
 uc32
 uc33
 uc34
 uc35
 uc36
 uc37
 uc38
 uc39
 uc40
 uc41
 uc42
 uc43
 uc44
 uc45
 uc46
 uc47
 uc48
 uc49
 uc50
 uc51
 uc52
 uc53
 uc54
 uc55
 uc56
 uc57
 uc58
 uc59
 ud0
 ud1
 ud2
 ud3
 ud4
 ud5
 ud6
 ud7
 ud8
 ud9
 ud10
 ud11
 ud12
 ud13
 ud14
 ud15
 ud16
 ud17
 ud18
 ud19
 ud20
 ud21
 ud22
 ud23
 ud24
 ud25
 ud26
 ud27
 ud28
 ud29
 ud30
 ud31
 ud32
 ud33
 ud34
 ud35
 ud36
 ud37
 ud38
 ud39
 ud40
 ud41
 ud42
 ud43
 ud44
 ud45
 ud46
 ud47
 ud48
 ud49
 ud50
 ud51
 ud52
 ud53
 ud54
 ud55
 ud56
 ud57
 ud58
 ud59
End
