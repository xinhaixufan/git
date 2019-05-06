//复杂链表深拷贝
//包含两个指针，next和random
/*
class Node {
public:
    int val;
    Node* next;
    Node* random;

    Node() {}

    Node(int _val, Node* _next, Node* _random) {
        val = _val;
        next = _next;
        random = _random;
    }
};
*/
Node* copyRandomList(Node* head)
{
    if(!head)return NULL;
    Node* p=head;

    while(p)//复制链表节点和next指针,如下格式：1->2->3->4 => 1->1->2->2->3->3->4->4
    {
        Node* q = new Node(p->val,NULL,NULL);
        q->next = p->next;
        p->next = q;
        p = q->next;
    }
    p=head;
    Node* q=p->next;
    while(p)//复制random指针
    {
        if(p->random)q->random=p->random->next;
        p=q->next;
        if(p) q=p->next;
    }
    p=head;
    Node* nhead=head->next;
    q=p->next;
    while(p)//拆分链表
    {
        p->next=q->next;
        if(q->next)q->next=q->next->next;
        p=p->next;
        q=q->next;
    }
    return nhead;
}
