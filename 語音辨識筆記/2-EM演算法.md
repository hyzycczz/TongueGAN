# EM演算法
以下是EM演算法的東西，也是GMM會用到的 (註：EM為 Expectiation-Maximization 最大期望演算法)
* 凸函數：
  如果f為實數x的函數，二階微分大於等於0，則f稱為凸函數，二階微分大於0稱為嚴格凸函數。
  ![](https://miro.medium.com/max/145/1*nnsFX0ho11XjGAFrddq_Og.png)
  如果f為向量x的函數，二階微分矩陣(Hessian Matrix, H)如果是半正定(determinant(H)≥0)，則f稱j為凸函數，如果determinant(H)>0稱為嚴格凸函數。
  ![](https://miro.medium.com/max/124/1*G7Gq6KbgvcBgMZ2PgJvang.png)
  
 * Jensen's inequality
機率論觀點，假設φ是凸函數，X是隨機變量，也就是說符合以下不等式的就是Jensen's inequality
  ![](https://miro.medium.com/max/154/1*Px33ltJXY9sEFC2Zyqyvvw.png)
  
  * EM演算法
以下有簡單的講解：
https://www.gushiciku.cn/pl/pVAN/zh-tw
* YT理論教學影片
https://www.youtube.com/playlist?list=PLyAft-JyjIYpno8IfZZS0mnxD5TYZ6BIc