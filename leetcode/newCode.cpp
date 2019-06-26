#include <iostream>
#include <stack>
#include <limits.h>
#include <vector>
#include <queue>
#include <algorithm>
#include <cstring>

using namespace std;

struct ListNode 
{
	int val;
	struct ListNode *next;
	ListNode(int x) :val(x), next(NULL) {}
};
struct TreeNode 
{
	int val;
	struct TreeNode *left;
	struct TreeNode *right;
	TreeNode(int x) :val(x), left(NULL), right(NULL) {}
};
struct RandomListNode 
{
    int label;
    struct RandomListNode *next, *random;
    RandomListNode(int x) : label(x), next(NULL), random(NULL) {}
};
struct TreeLinkNode 
{
    int val;
    struct TreeLinkNode *left;
    struct TreeLinkNode *right;
    struct TreeLinkNode *next;
    TreeLinkNode(int x) :val(x), left(NULL), right(NULL), next(NULL) {}
};
template <class T>
void printVec2(vector<vector<T>> &vec2)
{
	for(int i=0; i<vec2.size(); i++)
	{
		for(int j=0; j<vec2[i].size(); j++)
		{
			cout << vec2[i][j] << " ";
		}
		cout << endl;
	}
}

//合并两个有序链表
ListNode* Merge(ListNode* pHead1, ListNode* pHead2)
{
	if(pHead1 == NULL) return pHead2;
	if(pHead2 == NULL) return pHead1;

	ListNode* listN;
	if(pHead1->val > pHead2->val)
	{
		listN = pHead2;
		listN->next = Merge(pHead1, pHead2->next);
	}
	else
	{
		listN = pHead1;
		listN->next = Merge(pHead1->next, pHead2);
	}
	return listN;
}

//输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）

bool isSubTree(TreeNode* pRoot1, TreeNode* pRoot2)
{
	if(pRoot2 == NULL) return true;
	if(pRoot1 == NULL) return false;
	if(pRoot1->val == pRoot2->val)
	{
		return isSubTree(pRoot1->left, pRoot2->left) && isSubTree(pRoot1->right, pRoot2->right);
	}
	else 
	{
		return false;
	}

}
bool HasSubtree(TreeNode* pRoot1, TreeNode* pRoot2)
{
	if(pRoot1 == NULL || pRoot2 == NULL) return false;
	return isSubTree(pRoot1, pRoot2) || HasSubtree(pRoot1->left, pRoot2) || HasSubtree(pRoot1->right, pRoot2);
}

//操作给定的二叉树，将其变换为源二叉树的镜像
void Mirror(TreeNode *pRoot)
{
	if(pRoot == NULL) return;
	TreeNode* p;
	p = pRoot->left;
	pRoot->left = pRoot->right;
	pRoot->right = p;

	Mirror(pRoot->left);
	Mirror(pRoot->right);
}

//定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））
class Solution 
{
public:
    void push(int value) 
    {
		stack1.push(value);
    }
    void pop() 
    {
        stack1.pop();
    }
    int top() 
    {
        return stack1.top();
    }
    int min() 
    {
    	int minNum = INT_MAX;
    	while(!stack1.empty())
    	{
    		if(minNum > stack1.top()) minNum = stack1.top();
    		stack2.push(stack1.top());
    		stack1.pop();

    	}
    	while(!stack2.empty())
    	{
    		stack1.push(stack2.top());
    		stack2.pop();
    	}
        return minNum;
    }

private:
	stack<int> stack1;
	stack<int> stack2;
};

//压栈、弹栈顺序
bool IsPopOrder(vector<int> pushV,vector<int> popV)
{
	stack<int> tmp;
	int len = pushV.size();
	if(0 == len) return false;
	for(unsigned int i=0, j=0; i<len; )
	{
		tmp.push(pushV[i++]);
		while(j < popV.size() && tmp.top() == popV[j])
		{
			tmp.pop();
			j++;
		}
	}
	return tmp.empty();
}

//逐层打印二叉树
vector<int> PrintFromTopToBottom(TreeNode* root)
{
	vector<int> vec;
	if(root == NULL) return vec;
	queue<TreeNode*> que;
	que.push(root);

	while(!que.empty())
	{
		if(que.front()->left != NULL)
		que.push(que.front()->left);
		if(que.front()->right != NULL)
		que.push(que.front()->right);
		vec.push_back(que.front()->val);
		que.pop();
	}
	return vec;
}

//判断该数组是不是某二叉搜索树的后序遍历的结果。
bool VerifySquenceOfBST(vector<int> sequence)
{
	int len = sequence.size();
	if(len == 0) return true;
	int i=0;
	while(--len)
	{
		while(sequence[i++] < sequence[len]);
		while(sequence[i++] > sequence[len]);
		if(i < len) return false;
		i=0;
	}
	return true;
}
//二叉搜索树中和为某一值得所有路径

void dfs(TreeNode* root,int expectNumber, vector<vector<int> > vec2, vector<int> vec1)
{
	vec1.push_back(root->val);
	if(!root->left && !root->right)
	{
		if(root->val == expectNumber)
		{
			vec2.push_back(vec1);
		}
	}
	if(root->left)
	{
		dfs(root->left, expectNumber - root->val, vec2, vec1);
	}
	if(root->right)
	{
		dfs(root->right, expectNumber - root->val, vec2, vec1);
	}
	vec1.pop_back();
}
vector<vector<int> > FindPath(TreeNode* root,int expectNumber)
{
	vector<vector<int> > vec2;
	vector<int> vec1;
	if(root)
	{
		dfs(root, expectNumber, vec2, vec1);
	}
	return vec2;
}
//复杂链表复制
RandomListNode* Clone(RandomListNode* pHead)
{
	if(pHead == NULL) return NULL;
	RandomListNode* head, *p = pHead;

	while(p)
	{
		RandomListNode* q = new RandomListNode(p->label);
		q->next = p->next;
		p->next = q;
		p = q->next;
	}
	p = pHead;
	while(p)
	{
		if(p->random) p->next->random = p->random->next;
		p = p->next->next;
	}

    RandomListNode *tmp;
    RandomListNode *qq;
    qq = pHead->next;
    head = qq;
    tmp = pHead;
    while(tmp)
    {
        tmp->next = tmp->next->next;
        if(qq->next)
        {
            qq->next = qq->next->next;
        }
        qq = qq->next;
        tmp = tmp->next;
    }
	return head;
}

//将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向
TreeNode* Convert(TreeNode* pRootOfTree)
{

}

//输出字符串的所有排列
void Permu(string str, vector<string>& vecStr, int begin, int end)
{
	if(begin == end)
	{
		vecStr.push_back(str);
		return;
	}
	for(int i=begin; i<end; i++)
	{
		if(i != begin && str[i] == str[begin]) continue;
		swap(str[i], str[begin]);
		Permu(str, vecStr, begin+1, end);
	}
}
vector<string> Permutation(string str)
{
	vector<string> vecStr;
	int len = str.size();
	if(len == 0) return vecStr;
 	sort(str.begin(), str.end());
 	Permu(str, vecStr, 0, len);
 	return vecStr;
}

//数组中出现次数超过一半的数字

/*思路二：如果有符合条件的数字，则它出现的次数比其他所有数字出现的次数和还要多。
在遍历数组时保存两个值：一是数组中一个数字，一是次数。遍历下一个数字时，若它与之前保存的数字相同，则次数加1，
否则次数减1；若次数为0，则保存下一个数字，并将次数置为1。遍历结束后，所保存的数字即为所求。然后再判断它是否符合条件即可。
*/
int MoreThanHalfNum_Solution(vector<int> numbers)
{
	int len = numbers.size();
	if(len == 0) return 0;
	int result = numbers[0];
	int num =  1;

	for (int i = 1; i < len; ++i)
	{
		if(num == 0)
		{
			result = numbers[i];
			num = 1;
		}
		else if(result == numbers[i])
		{
			num ++;
		}
		else
		{
			num--;
		}
	}
	num = 0;
	for(int i=0; i<len; i++)
	{
		if(numbers[i] == result)
		{
			num ++;
		}
	}
	if(num > len/2) return result;
	else return 0;
}

//数组前k小的数字
//维护大小为k的最大堆，保存当前最小的k个数
vector<int> GetLeastNumbers_Solution(vector<int> input, int k)
{
	vector<int> vec;
	if(input.empty() || k==0 || k > input.size()) return vec;
	priority_queue<int> p;

	int i=0;
	while(i<k) p.push(input[i++]);
	while(i<input.size())
	{
		if(input[i] < p.top())
		{
			p.pop();
			p.push(input[i]);
		}
		i++;
	}
	while(!p.empty())
	{
		vec.push_back(p.top());
		p.pop();
	}
	return vec;
}
//连续子数组的最大和
int FindGreatestSumOfSubArray(vector<int> arr)
{
	int maxN = arr[0]; //最大值
	int sumN = arr[0]; //局部和
	for(int i=1; i<arr.size(); i++)
	{
		if(sumN >= 0)
		{
			sumN += arr[i];
		}
		else
		{
			sumN = arr[i];
		}
		if(sumN > maxN) maxN = sumN;
	}
	return maxN;
}

//第N个丑数
int GetUglyNumber_Solution(int N)
{
	if(N < 7) return N;
	vector<int> vec(N);
	vec[0] = 1;

	int t2=0, t3=0, t5=0;
	for(int i=1; i<N; i++)
	{
		vec[i] = min(vec[t2] * 2, min(vec[t3] * 3, vec[t5] * 5));
		if(vec[t2] * 2 == vec[i]) t2++;
		if(vec[t3] * 3 == vec[i]) t3++;
		if(vec[t5] * 5 == vec[i]) t5++;
		cout << "vec[" << i << "] is: " << vec[i] << endl; 
	}
	return vec[N-1];
}

//数组中的逆序对
//归并排序的改进

long long InversePairsCore(vector<int>& data, vector<int>& copy, int start, int end)
{
	if(start == end)
	{
		copy[start] = data[start];
		return 0;
	}
	int len = (end-start) / 2;
	long long left = InversePairsCore(copy, data, start, start + len);
	long long right = InversePairsCore(copy, data, start + len + 1, end);

	int i = start + len;
	int j = end;
	int indexCopy = end;
	long long count = 0;

	while(i >= start && j >= start + len + 1)
	{
		if(data[i] > data[j])
		{
			copy[indexCopy--] = data[i--];
			count = count + j - (start + len + 1) + 1;
		}
		else
		{
			copy[indexCopy--] = data[j--];
		}
	}
	while(i >= start) copy[indexCopy--] = data[i--];
	while(j >= start + len + 1) copy[indexCopy--] = data[j--];

	return left + right + count;
}
int InversePairs(vector<int> data)
{
	int len = data.size();
	if(len <= 0) return 0;
	vector<int> copy;
	for(int i=0; i<len; i++) copy.push_back(data[i]);
	long long count = InversePairsCore(data, copy, 0, len-1);
	return count % 1000000007;

}

//归并排序
void merger(vector<int>& vec, vector<int>& tmp, int start, int end, int mid)
{
	if(start == end)
	{
		return;
	}

	int i = start;
	int j = mid+1;
	int t = start;
	
	while(i <= mid && j <= end)
	{
		if(vec[i] < vec[j])
		{
			tmp[t++] = vec[i++];
		}
		else
		{
			tmp[t] = vec[j++];
			t++;
		}
	}
	while(i <= mid)
	{
		tmp[t++] = vec[i++];
	}
	while(j <= end)
	{
		tmp[t++] = vec[j++];
	}
	for(int i=0; i<end-start+1; i++) vec[start+i] = tmp[start+i];
}
void mergeSort(vector<int>& vec, vector<int>& tmp, int start, int end)
{
    if(start < end)
    {
        int m = start + (end-start)/2;
        mergeSort(vec, tmp, start, m);
        mergeSort(vec, tmp, m+1, end);
        merger(vec, tmp, start, end, m);
    }
    return ;
}

//把数组排列成最小的数
/*

 对vector容器内的数据进行排序，按照 将a和b转为string后
 若 a＋b<b+a  a排在在前 的规则排序,
 如 2 21 因为 212 < 221 所以 排序后为 21 2 
 to_string() 可以将int 转化为string
*/
bool cmp(int a, int b)
{
	string A = to_string(a) + to_string(b);
	string B = to_string(b) + to_string(a);
	return A < B;
}
string PrintMinNumber(vector<int> numbers)
{
	int len = numbers.size();
	string reStr = "";
	if(len == 0) return reStr;

	sort(numbers.begin(), numbers.end(), cmp);
	for(int i=0; i<len; i++)
	{
		reStr += to_string(numbers[i]);
	}
	return reStr;
}

//数字在排序数组中出现的次数
int bSearch(vector<int>& vec, int left, int right, int k)
{
	while(left <= right)
	{
		int mid = (right + left) / 2;
		if(vec[mid] == k) return mid;
		else if(vec[mid] < k) left = mid + 1;
		else right = mid - 1;
	}
	return -1;
}
int GetNumberOfK(vector<int> data ,int k)
{
	int len = data.size();
	int count = 1;
	int pos = bSearch(data, 0, len-1, k);
	int r = pos+1;
	int l = pos-1;
	if(pos == -1) return 0;
	while(r < len && data[r] == k)
    {
        r++;
        count++;
    }
    
	while(l >=0 && data[l] == k) 
    {
        l--;
        count++;
    }
	return count; 
}
//求二叉树的深度
int TreeDepth(TreeNode* pRoot)
{
	if(pRoot == NULL) return 0;
	return max(TreeDepth(pRoot->left), TreeDepth(pRoot->right)) + 1;
}
//判断二叉树是否是平衡二叉树
bool IsBalanced_Solution(TreeNode* pRoot) 
{
    if(pRoot == NULL) return true;
    if(abs(TreeDepth(pRoot->left) - TreeDepth(pRoot->right)) > 1) return false;
    else return IsBalanced_Solution(pRoot->left) && IsBalanced_Solution(pRoot->right);
}

// 一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字
void FindNumsAppearOnce(vector<int> data,int& num1,int& num2)
{
	int len = data.size();
	int n=0;
	int tmp = 1;
	vector<int> vec1, vec2;
	for(int i=0; i<len; i++)
	{
		n = n ^ data[i];
	}
	while((n & 1) == 0)
	{
		tmp = tmp << 1;
		n = n >> 1;
	}
	for(int i=0; i<len; i++)
	{
		if(data[i] & tmp != 0) vec1.push_back(data[i]);
		else vec2.push_back(data[i]);
	}
	int a=0, b=0;
	for(int i=0; i<vec1.size(); i++) 
	{
		a = a^vec1[i];
	}

			
	for(int i=0; i<vec2.size(); i++)
	{
		b = b^vec2[i];
	} 

	num1 = a;
	num2 = b;
	cout << "*num1: " << num1 << endl;
	return;
}
//和为S的连续正数序列
vector<vector<int> > FindContinuousSequence(int sum)
{
	vector<vector<int>> vec2;
	// vector<int> vec1;

	int l=1, r=1;
	int num = 0;
	while(r <= sum && l <= r)	
	{
		if(num < sum)
		{
			// cout << "num < sum" << endl;
			// cout << "l: " << l << ", r: " << r << endl;
			// cout << "num: " << num << endl;
			num += r;
			r++;
		}
		else if(sum == num)
		{
			// cout << "num = sum = " << num << endl;
			vector<int> vec1;
			for(int i=l; i<r; i++) vec1.push_back(i);
			vec2.push_back(vec1);
			num -= l;
			l++;
		}
		else
		{
			// cout << "num > sum" << endl;
			// cout << "l: " << l << ", r: " << r << endl;
			// cout << "num: " << num << endl;
			num -= l;
			l++;
		}
	}
	return vec2;
}
//和为S的两个数字
//输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。
vector<int> FindNumbersWithSum(vector<int> array,int sum)
{
	vector<int> vec;
	int l = 0, r = array.size() - 1;
	while(l < r)
	{
		if(array[l] + array[r] == sum) 
		{
			vec.push_back(array[l]);
			vec.push_back(array[r]);
			break;
		}
		else if(array[l] + array[r] > sum) r--;
		else l++;
	}
	return vec;
}
//左旋转字符串
string LeftRotateString(string str, int n) 
{
    if(n > str.size()) return "";
    string subStr1 = str.substr(0, n);
    string subStr2 = str.substr(n, str.size()-n);
    return subStr2 + subStr1;
}
//句子中单词反转
string ReverseSentence(string str) 
{
    vector<string> strvec;
    string restr = "";
    string tmp = "";
    if(str.size() == 0) return restr;
    for(int i=0; i<str.size(); i++)
    {
        if(str[i] != ' ') tmp += str[i];
        if(str[i] == ' ' || i == str.size()-1)
        {
            strvec.push_back(tmp);
            tmp = "";
        }
    }
	for(int i=strvec.size()-1; i>0; i--)
    {
        restr += strvec[i] + ' ';
    }
    restr += strvec[0];
    return restr;
}
//约瑟夫环
//f(n) = (f(n-1)+m)%n.
//f(1) = 0.
int LastRemaining_Solution(int n, int m)
{
	if(0 == n) return -1;
	int s = 0;
	for(int i=2; i<=n; i++)
	{
		s = (s+m)%i;
		cout << "s: " << s << endl;
	} 
	return s;
}
//求两个整数之和，要求在函数体内不得使用+、-、*、/四则运算符号。
int Add(int num1, int num2)
{
	while(num2)
	{
		int tmp = num1 ^ num2;
		num2 = (num2 & num1) << 1;
		num1 = tmp;
	}
	return num1;
}
//数组中第一个重复的数字
bool duplicate(int numbers[], int length, int* duplication)
{
	// vector<int> vec(length);
	// for(int i=0; i<length; i++)
	// {
	// 	vec[numbers[i]]++;
	// 	if(vec[numbers[i]] >= 2)
	// 	{
	// 		*duplication = numbers[i];
	// 		return true;
	// 	}
	// }
	// return false;
	for(int i=0; i<length; i++)
	{
		int index = numbers[i];
		if(index >= length)
		{
			index -= length;
		}
		if(numbers[index] >= length) 
		{
			*duplication = index;
			return true;
		}
		numbers[index] += length;
	}
	return false;
}

//构建乘积数组
vector<int> multiply(const vector<int>& A)
{
	vector<int> B(A.size());
	B[0] = 1;
	for(int i=1; i<A.size(); i++)
	{
		B[i] = B[i-1] * A[i];
	}
	int tmp = 1;
	for(int i = A.size()-1; i>=0; i--)
	{
		if(i == 0) B[i] = tmp;
		else
		{
			B[i] = tmp*B[i-1];
			tmp *= A[i];
		}
	}
	return B;
}
//正则表达式匹配
bool match(char* str, char* pattern)
{
	if(str[0] == '\0' && pattern[0] == '\0')
	{ 
		cout << "1" << endl;
		return true;
	}
	if(pattern[0] != '\0' && pattern[1] == '*')
	{
		cout << "2" << endl;
		// cout << str << endl;
		cout << pattern << endl;
		if(match(str, pattern + 2)) return true;
	}
	if(str[0] == pattern[0] || (str[0] != '\0' && pattern[0] == '.'))
	{
		cout << "3" << endl;
		cout << pattern << endl;
		if(match(str + 1, pattern + 1))
		{
			return true;
		}
		if(pattern[1] == '*' && match(str + 1, pattern))
		{
			return true;
		}
	}
	return false;
}
//表示数值的字符串
bool isNumeric(char* str)
{
	bool sign  = false, decimal = false, hasE = false;
	int len = strlen(str);
	for(int i=0; i<len; i++)
	{
		if(str[i] == 'e' || str[i] == 'E')
		{
			if(hasE || i == len -1) return false;
			hasE = true;
		}
		else if(str[i] == '+' || str[i] == '-')
		{
			if(sign && str[i-1] != 'e' && str[i] != 'E') return false;
			if(!sign && i>0 && str[i-1] != 'e' && str[i] != 'E' ) return false;
			sign = true;
		}
		else if(str[i] == '.')
		{
			if(hasE || decimal) return false;
			decimal = true;
		}
		else if(str[i] >'9' || str[i] < '0') return false;
	}
	return true;
}

//字符流中，第一个只出现一次的字符
/*
思路：时间复杂度O（1），空间复杂度O（n）
        1、用一个128大小的数组统计每个字符出现的次数
        2、用一个队列，如果第一次遇到ch字符，则插入队列；其他情况不在插入
        3、求解第一个出现的字符，判断队首元素是否只出现一次，如果是直接返回，否则删除继续第3步骤
*/
class Solution1
{
public:
  //Insert one char from stringstream
    void Insert(char ch)
    {
    	hash[ch - '\0'] += 1;
        if(hash[ch - '\0'] == 1) que.push(ch);
    }
  //return the first appearence once char in current stringstream
    char FirstAppearingOnce()
    {
    	while(!que.empty() && hash[que.front() - '\0'] > 1) que.pop();
    	if(!que.empty()) return que.front();
    	else return '#';
    }
private:
	queue<char> que;
	int hash[300] = {0};
};
//给一个链表，若其中包含环，请找出该链表的环的入口结点，否则，输出null
ListNode* EntryNodeOfLoop(ListNode* pHead)
{
	ListNode* slow = pHead;
	ListNode* fast = pHead;
	while(slow != NULL && fast != NULL)
	{
		slow = slow->next;
		if(fast->next != NULL) fast = fast->next->next;
		else return NULL;
		if(slow == fast) break;
	}
	int i=1;
	slow = slow->next;
	fast = fast->next->next;
	while(slow != fast)
	{
		i++;
		slow = slow->next;
		fast = fast->next->next;
	}
	slow = pHead, fast = pHead;
	while(i--) fast = fast->next;
	while(slow != fast)
	{
		slow = slow->next;
		fast = fast->next;
	}
	return slow;
}
//删除链表的重复节点
ListNode* deleteDuplication(ListNode* pHead)
{
	if(pHead == NULL) return NULL;
	ListNode* pre = NULL;
	ListNode* p = pHead;
	ListNode* q = NULL;
	while(p != NULL)
	{
		if(p->next != NULL && p->next->val == p->val)
		{
			q = p->next;
			while(q != NULL && q->next != NULL && q->next->val == p->val) q = q->next;
			if(p == pHead) pHead = q->next;
			else pre->next = q->next; 
			p = q->next;
		}
		else
		{
			pre = p;
			p = p->next;
		}
	}
	return pHead;
}
//给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。
TreeLinkNode* GetNext(TreeLinkNode* pNode)
{
	if(pNode == NULL) return NULL;
	if(pNode -> right != NULL)
	{
		pNode = pNode->right;
		while(pNode->left)
		{
			pNode = pNode->left;
		}
		return pNode;
	}
	while(pNode->next != NULL)
	{
		TreeLinkNode* tmp = pNode->next;
		if(tmp -> left == pNode) return tmp;
		pNode = pNode->next;
	}
	return NULL;
}
//判断一棵二叉树是不是对称的
bool isSym(TreeNode* left, TreeNode* right)
{
	if(left == NULL && right == NULL) return true;
	else if(left == NULL || right == NULL) return false;
	if(left->val == right->val)
	{
		return isSym(left->left, right->right) && isSym(left->right, right->left);
	}
	return false;
}
bool isSymmetrical(TreeNode* pRoot)
{
	if(pRoot == NULL) return true;
	return isSym(pRoot->left, pRoot->right);
}
//之字形打印二叉树
vector<vector<int> > Print1(TreeNode* pRoot) 
{
	vector<vector<int>> vec;
	stack<TreeNode*> sta1;
	stack<TreeNode*> sta2;
	if(!pRoot) return vec;
	sta1.push(pRoot);

	TreeNode * node;
	while(!sta1.empty() || !sta2.empty())
	{
		vector<int> ve1, ve2;
		while(!sta1.empty())
		{
		    node = sta1.top();
		    ve1.push_back(node->val);
	        if(node->left)
	        {
	        	sta2.push(node->left);
	        }
	        if(node->right)
	    	{
	            sta2.push(node->right);
	        }
	        sta1.pop();
        }

        if(!ve1.empty()) vec.push_back(ve1);

        while(!sta2.empty())
        {
            node = sta2.top();
            ve2.push_back(node->val);
            if(node->right)
            {
                sta1.push(node->right);
            }
            if(node->left)
            {
                sta1.push(node->left);
            }
            sta2.pop();
        }
        if(!ve2.empty()) vec.push_back(ve2);
    }
    return vec;
}
//二叉树按行打印
vector<vector<int> > Print(TreeNode* pRoot)
{
	vector<vector<int>> vec2d;
	if(pRoot == NULL) return vec2d;
	queue<TreeNode*> stack1;
	queue<TreeNode*> stack2;
	stack1.push(pRoot);
	while(!stack1.empty() || !stack2.empty())
	{
		vector<int> vec1;
		vector<int> vec2;
		while(!stack1.empty())
		{
			TreeNode* tmp = stack1.front();
			vec1.push_back(tmp->val);
			if(tmp->left)  stack2.push(tmp->left);
			if(tmp->right) stack2.push(tmp->right);
			stack1.pop();
		}
		if(!vec1.empty()) vec2d.push_back(vec1);

		while(!stack2.empty())
		{
			TreeNode* tmp = stack2.front();
			vec2.push_back(tmp->val);
			if(tmp->left)  stack1.push(tmp->left);
			if(tmp->right) stack1.push(tmp->right);
			stack2.pop();

		}
		if(!vec2.empty()) vec2d.push_back(vec2);

	}
	return vec2d;
}
//序列化和反序列化二叉树
class Solution2
{
public:
    char* Serialize(TreeNode *root) 
    {    
        
    }
    TreeNode* Deserialize(char *str) 
    {
    
    }
};
//滑动窗口的最大数
//矩阵中的路径

bool isHasPath(vector<vector<char>>& vec2d, vector<vector<int>>& flag, int rows, int cols, char* str)
{
	cout << "str = " << str << endl;
	if(*str == '\0') return true;
	if(rows < 0 || rows >= vec2d.size() || cols < 0 || cols >= vec2d[0].size()) return false;
	
	if(str[0] == vec2d[rows][cols] && flag[rows][cols] == 0)
	{
		cout << rows << endl << cols << endl;
		flag[rows][cols] = 1;
		return isHasPath(vec2d, flag, rows-1, cols, str+1)||isHasPath(vec2d, flag, rows+1, cols, str+1)||isHasPath(vec2d, flag, rows, cols-1, str+1)||isHasPath(vec2d, flag, rows, cols+1, str+1);
	}
	return false;
}

bool hasPath(char* matrix, int rows, int cols, char* str)
{

	if(*str == '\0') return true;
	if(*matrix == '\0') return false;
	vector<vector<char>> vec2d;
	int count = 0;
	for(int i=0; i<rows; i++)
	{
		vector<char> vec;
		for(int j=0; j<cols; j++)
		{
			vec.push_back(matrix[count++]);
		}
		vec2d.push_back(vec);
	}
	
	for(int i=0; i<vec2d.size(); i++)
	{
		for(int j=0; j<vec2d[i].size(); j++)
		{
			vector<vector<int>> flag(rows, vector<int>(cols, 0));
			if(isHasPath(vec2d, flag, i, j, str)) return true;
		}
	}
	return false;
}
//机器人的运动范围
class Solution3 
{
public:
    int movingCount(int threshold, int rows, int cols)
    {
        vector<vector<bool>> vec2d(rows, vector<bool>(cols, false));
        printVec2(vec2d);
        return wfs(threshold, rows, cols, 0, 0, vec2d);
    }
private:
	int add(int n)
	{
		int count = 0;
		while(n)
		{
			count += n%10;
			n /= 10;
		}
		return count;
	}
	int wfs(int th, int rows, int cols, int i, int j, vector<vector<bool>>& flag)
	{
		if(i<0 || i>=rows || j<0 || j>=cols || add(i)+add(j)>th || flag[i][j]) return 0;
		flag[i][j] = true;
		return wfs(th, rows, cols, i+1, j, flag) + wfs(th, rows, cols, i-1, j, flag) + wfs(th, rows, cols, i, j-1, flag) + wfs(th, rows, cols, i, j+1, flag) + 1;
	}
};
int main()
{
	Solution3 su;
	cout << su.movingCount(18, 4, 4) << endl;
	return 0;
}